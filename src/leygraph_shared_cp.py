
from src.model_par import *
from src.user_cp import *
from src.ley_agg import *
from src.ley_module import *
from src.origin_module import *


def LeyGraph_Shared_Cp(M1, M2, User_defined=0):
    if User_defined:     
        #Modify according to your own requirements
        sh_cp = User_Cp(M1.in_channels, M1.hidden_channels)
        mM1 = Ley_Module(M1.model_type, M1.in_channels, M1.hidden_channels, M1.out_channels, M1.num_layers)
        mM2 = Ley_Module(M2.model_type, M2.in_channels, M2.hidden_channels, M2.out_channels, M2.num_layers)
        cp_exist = 1
    else: #user_defined==0
        if (M1.model_type=="sage" and M2.model_type=="gin") or (M1.model_type=="gin" and M2.model_type=="sage"):
            sh_cp = Ley_Agg(M1.in_channels, M1.hidden_channels)
            mM1 = Ley_Module(M1.model_type, M1.in_channels, M1.hidden_channels, M1.out_channels, M1.num_layers)
            mM2 = Ley_Module(M2.model_type, M2.in_channels, M2.hidden_channels, M2.out_channels, M2.num_layers)
            cp_exist = 1
        else:
            sh_cp = 0
            mM1 = Origin_Module(M1.model_type, M1.in_channels, M1.hidden_channels, M1.out_channels, M1.num_layers)
            mM2 = Origin_Module(M2.model_type, M2.in_channels, M2.hidden_channels, M2.out_channels, M2.num_layers)
            cp_exist = 0
    return mM1, mM2, cp_exist, sh_cp

