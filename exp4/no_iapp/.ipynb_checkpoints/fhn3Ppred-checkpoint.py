
from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.fourier_net import FourierNetArch
from modulus.models.siren import SirenArch
from modulus.models.modified_fourier_net import ModifiedFourierNetArch
from modulus.models.dgm import DGMArch

from sympy import Symbol, Eq
from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Point1D
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from modulus.eq.pde import PDE
from modulus.geometry import Parameterization
from sympy import Symbol, Eq, Abs, tanh, Or, And
from modulus.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
from modulus.solver import SequentialSolver
from modulus.domain.monitor import PointwiseMonitor
from modulus.models.deeponet import DeepONetArch
from modulus.domain.constraint.continuous import DeepONetConstraint
from modulus.models.moving_time_window import MovingTimeWindowArch
from modulus.domain.monitor import Monitor
from modulus.domain.constraint import Constraint
from modulus.graph import Graph
from modulus.key import Key
from modulus.constants import TF_SUMMARY
from modulus.distributed import DistributedManager
from modulus.utils.io import dict_to_csv, csv_to_dict
from modulus.domain.inferencer.pointwise import PointwiseInferencer as PointwiseInferencer
from modulus.loss.loss import CausalLossNorm

  
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.cuda.profiler as profiler
import torch.distributed as dist
from termcolor import colored, cprint
from copy import copy
from operator import add
from omegaconf import DictConfig, OmegaConf
import hydra
import itertools
from collections import Counter
from typing import Dict, List, Optional
import logging
from contextlib import ExitStack
from typing import List, Union, Tuple, Callable


from modulus.domain.constraint import Constraint
from modulus.domain import Domain
from modulus.loss.aggregator import Sum
from modulus.utils.training.stop_criterion import StopCriterion
from modulus.constants import TF_SUMMARY, JIT_PYTORCH_VERSION
from modulus.hydra import (
    instantiate_optim,
    instantiate_sched,
    instantiate_agg,
    add_hydra_run_path,
)
from modulus.distributed.manager import DistributedManager
 

    
t_max = 50.0
n_w=1
t_w= t_max/n_w
BTZ=2000

def print_folder_contents(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    print(f"Contents of folder '{folder_path}':")
    for item in os.listdir(folder_path):
        print(item)

def generateExactSolution(t,dt,x0,w0,rate,P,begin,end):
    
    
    n2=int(t/(dt))+2
    n = int((end-begin)/(dt*rate))
    Sol=np.zeros((n,3))
  
    Sol2=np.zeros((n2,2))
    Sol2[0]=x0,w0
    T=0
    k=0
    while(k<n2-1):
        x,w=Sol2[k]
        Sol2[k+1]=10*(x*(x-0.4)*(1-x)-w + P)*dt+  x, 0.2*(x*0.2-0.8*w)*dt +w
 
        if ((k*dt==begin or ((k+1)%rate == 0 and k*dt>=begin and k*dt<=end))and T<n):
          
           
            Sol[T] = Sol2[k][0],Sol2[k][1] , k*dt
            T=T+1
        
        k=k+1
        if(k*dt > end):
            break
    return Sol

def generateValidator(i,nodes,data_folder="../.././validation/"):
    # Read data arrays from CSV files
    
    print_folder_contents(data_folder)
    T = np.load(data_folder + "T.npy")
    K = np.load(data_folder + "K.npy")
    U = np.load(data_folder + "U.npy")
    V = np.load(data_folder + "V.npy")
    SOLs = np.load(data_folder + "SOLs.npy")
    SOLw = np.load(data_folder + "SOLw.npy")

    t = np.expand_dims(T, axis=-1)
    k = np.expand_dims(K, axis=-1)
    u = np.expand_dims(U, axis=-1)
    Solx = np.expand_dims(SOLs, axis=-1)
    Solw = np.expand_dims(SOLw, axis=-1)
    v = np.expand_dims(V, axis=-1)

    print("Added validation of shape", np.shape(Solx))
    invar_numpy = {
        "t": t,
        "K": k,
        "V": v,
        "U": u,
    }

    outvar_numpy = {
        "x1": Solx,
        "w":Solw
    }
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=1024,plotter=None
    )
    return validator



def generateDataC(i,nodes,data_folder="../.././treino1/"):
    # Read data arrays from CSV files
    
    print_folder_contents(data_folder)

    T = np.load(data_folder + "T.npy")
    K = np.load(data_folder + "K.npy")
    U = np.load(data_folder + "U.npy")
    V = np.load(data_folder + "V.npy")
    SOLs = np.load(data_folder + "SOLs.npy")
    SOLw = np.load(data_folder + "SOLw.npy")

    t = np.expand_dims(T, axis=-1)
    k = np.expand_dims(K, axis=-1)
    u = np.expand_dims(U, axis=-1)
    Solx = np.expand_dims(SOLs, axis=-1)
    Solw = np.expand_dims(SOLw, axis=-1)
    v = np.expand_dims(V, axis=-1)

    invar_numpy = {
        "t": t,
        "K": k,
        "V": v,
        "U": u,
    }
    print("Added data constraint of shape", np.shape(Solx))

    outvar_numpy = {
        "x1": Solx,
        "w":Solw
    }
    constraint = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar=invar_numpy,
        outvar=outvar_numpy,
        batch_size=BTZ,
        lambda_weighting=None
    )

    return constraint


class SpringMass(PDE):
    name = "SpringMass"

    def __init__(self):

        t = Symbol("t")
        K = Symbol("K")
       
        input_variables = {"t": t,"K":K}

        x = Function("x1")(*input_variables)
        w= Function("w")(*input_variables)
        self.equations = {}
        self.equations["ode_x1"] =10*(x*(x-0.4)*(1-x)-w) -x.diff(t)
        self.equations["ode_w"]  =K*(x*0.2-0.8*w) -w.diff(t)
        
@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:


    # make list of nodes to unroll graph on
    sm = SpringMass()
    sm.pprint()
    #sm_net = FullyConnectedArch(
    #    input_keys=[Key("t"), Key("K")],
    #    output_keys=[Key("x1")],
    #)
    #nodes = sm.make_nodes() + [
    #    sm_net.make_node(name="network")
    #]


    
    # make list of nodes to unroll graph on
    sm = SpringMass()
    sm.pprint()
    #sm_net = FullyConnectedArch(
    #    input_keys=[Key("t"), Key("K")],
    #    output_keys=[Key("x1")],
    #)
    #nodes = sm.make_nodes() + [
    #    sm_net.make_node(name="network")
    #]

    
    
    flow_net = FullyConnectedArch(
            input_keys=[Key("t"), Key("U"),Key("V"),Key("K") ],
            output_keys=[Key("x1"),Key("w")],
            layer_size=64,
            nr_layers=8,
        )

    

    #time_window_net = MovingTimeWindowArch(flow_net, t_w)

    nodes = sm.make_nodes() +[flow_net.make_node(name="network")]


    for node in nodes:
        print(node.__str__())
   
    # add constraints to solver
    # make geometry
    geo = Point1D(0)
    
    t_symbol = Symbol("t")
    x_symbol = Symbol("x1")
    k_symbol= Symbol("K")
    v_symbol= Symbol("V")
    u_symbol= Symbol("U")
    
    time_range = {t_symbol: (0,t_w )}
    k_range= {k_symbol:(0.08,0.012)}
    v_range= {v_symbol:(0,0.12)}
    u_range= {u_symbol:(0,1)}

    tr = {t_symbol: (0, t_w)}

    # make domain
        # make initial condition domain
    ic_domain = Domain("initial_conditions")

  
    # initial conditions
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"x1": u_symbol,"w":v_symbol},
        batch_size=BTZ,
        parameterization={**{t_symbol:0},**k_range,**v_range,**u_range},
        lambda_weighting={
            "x1": 1000,# + 1000*x_symbol.diff(t_symbol)*x_symbol.diff(t_symbol),
            "w": 1000 #+ 1000*x_symbol.diff(t_symbol)*x_symbol.diff(t_symbol)
        },
        
        
        quasirandom=True,
    )

    ic_domain.add_constraint(IC, name="IC")
    
        # solve over given time period
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0,"ode_w":0.0},
        batch_size=BTZ,
        parameterization={**tr,**k_range,**v_range,**u_range},
        #criteria=And(t_symbol > 0, t_symbol < 3),
        lambda_weighting={
            "ode_x1": 100,# + 1000*x_symbol.diff(t_symbol)*x_symbol.diff(t_symbol),
            "ode_w":100 #+ 1000*x_symbol.diff(t_symbol)*x_symbol.diff(t_symbol)
        },
        quasirandom=True,
    )
    ic_domain.add_constraint(interior, name="interior")
    
    
       # solve over given time period
    interior2 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0,"ode_w":0.0},
        batch_size=BTZ,
        parameterization={**tr,**{u_symbol:(0.45,0.65)},**{v_symbol:(0.0,0.12)},**k_range},
        #criteria=And(t_symbol > 0, t_symbol < 3),
        lambda_weighting={
            "ode_x1":100,# + 1000*x_symbol.diff(t_symbol)*x_symbol.diff(t_symbol),
            "ode_w": 100 #+ 1000*x_symbol.diff(t_symbol)*x_symbol.diff(t_symbol)
        },
        quasirandom=True,
    )
   #ic_domain.add_constraint(interior2, name="interiorTr")
        
        
    data_folder="../.././extrapol/"
    T = np.load(data_folder + "T.npy")
    K = np.load(data_folder + "K.npy")
    U = np.load(data_folder + "U.npy")
    V = np.load(data_folder + "V.npy")
    SOLs = np.load(data_folder + "SOLs.npy")
    SOLw = np.load(data_folder + "SOLw.npy")

    t = np.expand_dims(T, axis=-1)
    k = np.expand_dims(K, axis=-1)
    u = np.expand_dims(U, axis=-1)
    Solx = np.expand_dims(SOLs, axis=-1)
    Solw = np.expand_dims(SOLw, axis=-1)
    v = np.expand_dims(V, axis=-1)

    print("Added monitor of shape", np.shape(Solx))
    invar_numpy = {
        "t": t,
        "K": k,
        "V": v,
        "U": u,
        "True_u":Solx,
        "True_w":Solw
    }
    

    
    extrapol_val = PointwiseMonitor(
         invar_numpy,
         output_names=["x1", "w"],
         metrics={
          
             "Extra_error_mean": lambda var: torch.mean(
                
                 torch.abs((torch.abs(var["True_u"]) - torch.abs(var["x1"])))
             ),
             
             "Extra_error_max": lambda var: torch.max(
                
                 torch.abs((torch.abs(var["True_u"]) - torch.abs(var["x1"])))
             ),
         },
         nodes=nodes,
     )
    ic_domain.add_monitor(extrapol_val)
 
    
    
    print("Added monitor of shape", np.shape(Solx))

    
    
    data_folder="../.././interpol/"
    T = np.load(data_folder + "T.npy")
    K = np.load(data_folder + "K.npy")
    U = np.load(data_folder + "U.npy")
    V = np.load(data_folder + "V.npy")
    SOLs = np.load(data_folder + "SOLs.npy")
    SOLw = np.load(data_folder + "SOLw.npy")

    t = np.expand_dims(T, axis=-1)
    k = np.expand_dims(K, axis=-1)
    u = np.expand_dims(U, axis=-1)
    Solx = np.expand_dims(SOLs, axis=-1)
    Solw = np.expand_dims(SOLw, axis=-1)
    v = np.expand_dims(V, axis=-1)

    print("Added validation of shape", np.shape(Solx))
    invar_numpy = {
        "t": t,
        "K": k,
        "V": v,
        "U": u,
        "True_u":Solx,
        "True_w":Solw
    }
    

    
    interpol_val = PointwiseMonitor(
         invar_numpy,
         output_names=["x1", "w"],
         metrics={
          
             "Inter_error_mean": lambda var: torch.mean(
                
                 torch.abs((torch.abs(var["True_u"]) - torch.abs(var["x1"])))
             ),
             
             "Inter_error_max": lambda var: torch.max(
                
                torch.abs((torch.abs(var["True_u"]) - torch.abs(var["x1"])))
             ),
         },
         nodes=nodes,
     )
    ic_domain.add_monitor(interpol_val)
 
    
    

    
    dom=[]
    dom.append((1,ic_domain))
    print(cfg)
    # make solver
    #slv = Solver(cfg, domain)
    #print(domains)
    i=0
    for a,d in dom:
      #  print(d)
      #  print(d.name)
#        d.add_inferencer(generateValidator(i,nodes))
        d.add_validator(generateValidator(i,nodes))
        d.add_constraint(generateDataC(i,nodes,data_folder="../.././treino1/"),"data1")
       # d.add_constraint(generateDataC(i,nodes,data_folder="../.././treino2/"),"data2")
        d.add_constraint(generateDataC(i,nodes,data_folder="../.././treino3/"),"data3")
       

        i=i+1
    
     
    slv = SequentialSolver(
        cfg,
        dom,

    )
    slv.solve()

        
        
        
if __name__ == "__main__":
    run()
