import numpy as np
import components

def update_offset(dend,update,offmax,traj):
    # print("here")


    # if dend.outgoing[0][1] < 0: 
    #     # print("negative update:",dend.name)
    #     update*=-1

    dend.flux_offset += update
    if offmax==0: offmax = dend.phi_th
    if dend.flux_offset > 0:
        dend.flux_offset = np.min([dend.flux_offset, offmax])
    elif dend.flux_offset < 0:
        dend.flux_offset = np.max([dend.flux_offset, -1*offmax])
    if traj==True: dend.update_traj.append(dend.flux_offset)

def update_connection_strength(dend,update,offmax,traj):
    pass

def symmetric_udpater(error,eta,dend,offmax,layers,traj):
    """
    Try this for synaptic layer only
    Play with zero-signal update coefficient
    """
    
    if dend.loc[0]==layers-1:
        if dend.outgoing[0][1] < 0: 
            update_sign = -1
        else:
            update_sign = 1

        if np.mean(dend.signal) > 0:
            update = np.mean(dend.signal)*error*eta*update_sign

        else: 
            update = 0
            # update = error*eta*update_sign*-1*.3
    else:
        update = np.mean(dend.signal)*error*eta

    update_offset(dend,update,offmax,traj)

def choosing_udpater(error,eta,dend,offmax,layers,traj):
    """

    """
    # print("chooser")
    if dend.loc[0]==layers-1:
        if dend.outgoing[0][1] < 0 and error<0: 
            update = np.mean(dend.signal)*error*eta*-1
        elif dend.outgoing[0][1] > 0 and error>0: 
            update = np.mean(dend.signal)*error*eta
        else: 
            update = 0
    else:
        update = np.mean(dend.signal)*error*eta

    update_offset(dend,update,offmax,traj)

def splitting_udpater(error,eta,dend,offmax,traj):
    """
    """
    # # print("splitting update")
    # if dend.outgoing[0][1] < 0: 
    #     update = np.mean(dend.signal)*error*eta*-1
    # elif dend.outgoing[0][1] > 0: 
    #     update = np.mean(dend.signal)*error*eta

    update = np.mean(dend.signal)*error*eta
    dend.flux_offset += update

    if np.sign(dend.flux_offset) != np.sign(dend.outgoing[0][1]):
        # print("Flip",dend.flux_offset,dend.outgoing[0][1])
        dend.outgoing[0][1] = dend.outgoing[0][1]*-1
    # dend.update_traj.append(dend.flux_offset)
    # update_offset(dend,update,offmax)


def make_update(node,error,eta,offmax,updater,traj=False):
    for i,dend in enumerate(node.dendrite_list):
        if np.any(dend.flux>0.5):  dend.high_roll += 1
        if np.any(dend.flux<-0.5): dend.low_roll  += 1

        if traj==True and not hasattr(dend,'update_traj'): 
            dend.update_traj = [dend.flux_offset]
            # print(dend.update_traj)
        if (not isinstance(dend,components.Refractory) 
            and not isinstance(dend,components.Soma)):
            if hasattr(dend,'update'):
                if dend.update==True:
                    if updater == 'symmetric':
                        symmetric_udpater(error,eta,dend,offmax,node.layers,traj)
                    elif updater == 'classic':
                        update = np.mean(dend.signal)*error*eta
                        update_offset(dend,update,offmax,traj)
                    elif updater == 'chooser':
                        choosing_udpater(error,eta,dend,offmax,node.layers,traj)
                    elif updater == 'splitter':
                        splitting_udpater(error,eta,dend,offmax,traj)

            else:
                if updater == 'symmetric':
                    symmetric_udpater(error,eta,dend,offmax,node.layers,traj)
                elif updater == 'classic':
                    update = np.mean(dend.signal)*error*eta
                    update_offset(dend,update,offmax,traj)
                elif updater == 'chooser':
                    choosing_udpater(error,eta,dend,offmax,node.layers,traj)
                elif updater == 'splitter':
                    splitting_udpater(error,eta,dend,offmax,traj)

                # update_offset(dend,error,eta,offmax)

def backpath(node,error,eta,offmax,chooser):
    
    soma = node.dend_soma
    if not hasattr(soma,'update_traj'): soma.update_traj = []

    if np.any(np.mean(soma.signal)>0): ds = 1
    else: ds = 0
    update = ds*error*eta
    
    # update = np.mean(soma.signal)*error*eta
    # if update < 0: update*=0.8
        # update = np.random.rand()*.1 #*np.random.choice([-1,1], p=[.5,.5], size=1)[0]
        # print(soma.name,update)
    # if node.name == 'node_z':print(node.name, error, np.mean(soma.signal), update)
    # update_offset(soma,update,offmax)
    soma.update_traj.append(update)
    
    for dend in node.dendrite_list[2:]:
        if np.any(dend.flux>0.5):  dend.high_roll += 1
        if np.any(dend.flux<-0.5): dend.low_roll  += 1
        if not hasattr(dend,'update_traj'): dend.update_traj = []
        # print(f"{dend.name} -- {dend.outgoing[0][0].name}")

        if np.any(np.mean(dend.signal)>0): ds = 1
        else: ds = 0

        update = ds*dend.outgoing[0][0].update_traj[-1]#*dend.outgoing[0][1] #*eta

        # update = np.mean(dend.signal)*dend.outgoing[0][0].update_traj[-1]*dend.outgoing[0][1]
        # if update < 0: update*=0.8
        # if node.name == 'node_z':
        #     print(
        #         f"{dend.name} -- {dend.outgoing[0][0].name} -- {update} -- {dend.outgoing[0][0].update_traj[-1]} -- {dend.outgoing[0][1]}"
        #         )
        
        update_offset(dend,update,offmax)