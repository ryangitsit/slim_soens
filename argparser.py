import argparse

def setup_argument_parser():
    '''
    Command line arguments to run with slim_soens
    '''  

    parser = argparse.ArgumentParser()
    
    parser.add_argument( "--patterns"          ,   type = int,   default = 12             )
    parser.add_argument( "--runs"              ,   type = int,   default = 1000           )
    parser.add_argument( "--duration"          ,   type = int,   default = 250            )
    parser.add_argument( "--print_mod"         ,   type = int,   default = 50             )
    parser.add_argument( "--plotting"          ,   type = bool,  default = False          )
    parser.add_argument( "--realtimeplt"       ,   type = bool,  default = False          )
    parser.add_argument( "--printing"          ,   type = bool,  default = True           )
    parser.add_argument( "--plot_trajectories" ,   type = bool,  default = True           )
    parser.add_argument( "--print_rolls"       ,   type = bool,  default = False          )
    parser.add_argument( "--update_type"       ,   type = str,   default = 'arbor'        )
    parser.add_argument( "--eta"               ,   type = float, default = 0.005          )
    parser.add_argument( "--fan_fact"          ,   type = int,   default = 2              )
    parser.add_argument( "--max_offset"        ,   type = float, default = .4             )
    parser.add_argument( "--target"            ,   type = int,   default = 2              )
    parser.add_argument( "--offset_radius"     ,   type = float, default = 0              )
    parser.add_argument( "--mutual_inh"        ,   type = float, default = -0.75          )
    parser.add_argument( "--doubled"           ,   type = bool,  default = False          )
    parser.add_argument( "--weight_type"       ,   type = str,   default = "double_dends" )
    parser.add_argument( "--updater"           ,   type = str,   default = "chooser"      )

    return parser.parse_args()