from __future__ import division, print_function
import run_LFP
import sKCSD3D
import utility_functions as utils
import sys
import os
import loadData as ld

if __name__ == '__main__':
    fname_base = "ball_stick_"
    paths = []
    for rownb in [8,32,128]:
        fname = fname_base+str(rownb)
        c = run_LFP.CellModel(morphology=1,cell_name=fname,colnb=1,rownb=rownb,xmin=-100,xmax=600,ymin=0,ymax=200)
        c.simulate()
        c.save_skCSD_python()
    
        data_dir = c.return_paths_skCSD_python()
    
        data = ld.Data(data_dir)
        scaling_factor = 100
        ele_pos = data.ele_pos
        pots = data.LFP[:,:200]
        params = {}
        morphology = data.morphology
        print(morphology)
        xmin = morphology[:,2].min()-morphology[:,5].max()
        xmax = morphology[:,2].max()+morphology[:,5].max()
        ymin = morphology[:,3].min()-morphology[:,5].max()
        ymax = morphology[:,3].max()+morphology[:,5].max()
        zmin = morphology[:,4].min()-morphology[:,5].max()
        zmax = morphology[:,4].max()+morphology[:,5].max()
        print(xmin,xmax,ymin,ymax,zmin,zmax)
        gdx = (xmax-xmin)/10
        gdy = (ymax-ymin)/10
        gdz = (zmax-zmin)/100
        k = sKCSD3D.sKCSD3D(ele_pos, pots,morphology, gdx=gdx, gdy=gdy, gdz=gdz, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, n_src_init=10, src_type='gauss_lim')
        #k.cross_validate()
    
        if sys.version_info >= (3, 0):
            path = os.path.join(data_dir,"preprocessed_data/Python_3")
        else:
            path = os.path.join(data_dir,"preprocessed_data/Python_2")
            
        if not os.path.exists(path):
            print("Creating",path)
            os.makedirs(path)
        
        utils.save_sim(path,k)
        paths.append(path)
