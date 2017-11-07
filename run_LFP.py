from __future__ import division, print_function
import numpy as np
import sys
import os
import random
import LFPy

import argparse
"""Electrode grid is 2D. If z is the zero dimention x=x, y=y. If x is the zero dimension x=0, y=x, z=y. If y is the zero dimention, z=x & x=y"""

"""cell_types = {'Ballstick:1, 'Y_shaped':2, 'Morpho1':3, 'Agasbogas':4, 'Mainen':5, 'User_defined:6', 'Gang_Simple':7, 'Domi':8}
electrode_orientation = {'x':1, 'y':2, 'z':3}
electrode_distribute = {'Grid':1, 'Random':2, 'Hexagonal':3, 'Domi':4}
LFPy_sim = {'Random':1, 'Y_symmetric':2, 'Mainen':3, 'Oscill':4, 'Const':5 }

"""
class CellModel():
    MORPHOLOGY_FILES = {
        1:"morphology/ballstick.swc",
        2:"morphology/villa.swc",
        3:"morphology/morpho1.swc",
        4:"morphology/neuron_agasbogas.swc",
        5:"morphology/Mainen_swcLike.swc",
        6:"morphology/retina_ganglion.swc",
        7:"morphology/Badea2011Fig2Du.CNG.swc",
        8:"morphology/DomiCell.swc",
        
    }
    CELL_PARAMETERS = {         
   

	'Ra': 123,
        'tstartms' : 0.,                 # start time of simulation, recorders start at t=0
        'tstopms' : 850.,                   # stop simulation at 200 ms. 
	'passive' : True,
    	'v_init' : -65,             # initial crossmembrane potential
    	'e_pas' : -65,              # reversal potential passive mechs
	'nsegs_method' :  'fixed_length',
	'max_nsegs_length':10, 
        'custom_code'  : [], # will run this file
    }

    SYNAPSE_PARAMETERS = { #idx to be set later
        'e' : 0.,                   # reversal potential
        'syntype' : 'ExpSyn',       # synapse type
        'tau' : 2.,
        'weight' : .04,            # syn. weight 
        'record_current' : True,
    }

    SIMULATION_PARAMETERS = {
	'rec_imem' : True,  # Record Membrane currents during simulation
	'rec_isyn' : True,  # Record synaptic currents
    }
    ELECTRODE_PARAMETERS = {
        'method' : 'linesource'
    }
    POINT_PROCESS = {
        'idx':0.,
        'pptype':'IClamp',
        
        }
    def __init__(self,**kwargs):
        
        self.cell_parameters = self.CELL_PARAMETERS.copy()
        self.synapse_parameters = self.SYNAPSE_PARAMETERS.copy()
        self.simulation_parameters = self.SIMULATION_PARAMETERS.copy()
        self.electrode_parameters = self.ELECTRODE_PARAMETERS.copy()
        self.point_process = self.POINT_PROCESS.copy()
        
        self.cell_name = kwargs.get('cell_name','cell_1')
        self.path = kwargs.get('path','simulation')
        self.stimulus = kwargs.get('stimulus','random')
        eldistribute = kwargs.get('electrode_distribution',1)
        orientation = kwargs.get('electrode_orientation',2)
        colnb = kwargs.get('colnb',4)
        rownb = kwargs.get('rownb',4)
        xmin = kwargs.get('xmin',0)
        xmax = kwargs.get('xmax',200)
        ymin = kwargs.get('ymin',-100)
        ymax = kwargs.get('xmax',-500)
        cell_electrode_dist = kwargs.get('electrode_distance',50)
        triside = kwargs.get('triside',19)
        ssNB = kwargs.get('seed',123456)
        custom_code = kwargs.get('custom_code','morphology/active.hoc')
        morphology = kwargs.get('morphology',1)
        self.sigma = kwargs.get('sigma',.5)
        self.n_pre_syn = kwargs.get('n_presyn',1000)
        self.n_synapses = kwargs.get('n_syn', 1000)
        self.new_path = os.path.join(self.path, self.cell_name)
        np.random.seed(ssNB)        

        if morphology in range(1,9):
            morphology = self.MORPHOLOGY_FILES[morphology]
        
        self.make_cell(morphology,custom_code)
        self.setup_LFPy_2D_grid(eldistribute,orientation,colnb,rownb,xmin,xmax,ymin,ymax,cell_electrode_dist,triside,ssNB)

        self.add_electrodes()
        
    def stationary_poisson(self,nsyn,lambd,tstart,tstop):
        ''' Generates nsyn stationary possion processes with rate lambda between tstart and tstop'''
        interval_s = (tstop-tstart)*.001
        spiketimes = []
        for i in range(nsyn):
            spikecount = np.random.poisson(interval_s*lambd)
            spikevec = np.empty(spikecount)
            if spikecount==0:
                spiketimes.append(spikevec)
            else:
                spikevec = tstart + (tstop-tstart)*np.random.random(spikecount)
                spiketimes.append(np.sort(spikevec)) #sort them too!
        return spiketimes

    def make_cell(self, morphology, custom_code=[]):
        self.cell_parameters['morphology'] = morphology

        if not morphology.endswith('.hoc'):
            self.cell_parameters['custom_code'].append('morphology/active.hoc')
        for code in custom_code:
            self.cell_parameters['custom_code'].append(custom_code)

        self.cell = LFPy.Cell(**self.cell_parameters)
        #self.cell.set_pos(xpos = LFPy.cell.neuron.h.x3d(0) , ypos = LFPy.cell.neuron.h.y3d(0) , zpos = LFPy.cell.neuron.h.z3d(0))
        return self.cell

    def save_morphology_to_file(self):
        segments = self.cell.get_idx()
        nseg = len(segments)
        segments_dict = {}
        for section in self.cell.allsecnames:
            segment = self.cell.get_idx(section=section)
            for seg in segment:
                segments_dict[seg] = section
 
        morphology = np.zeros((nseg,7))
        morph2 = np.zeros((nseg,7))
        coords = np.array((self.cell.xstart, self.cell.ystart, self.cell.zstart)).T
        ends =  np.array((self.cell.xend, self.cell.yend, self.cell.zend)).T
        sections = self.cell.allsecnames
        segdiam = self.cell.diam
 
        for seg in segments:
            
            number = seg + 1
            morphology[seg,0] = number

            if 'soma' in segments_dict[seg]:
                morphology[seg,1] = 1
            elif 'dend' in segments_dict[seg]:
                morphology[seg,1] = 3
            elif 'apic' in segments_dict[seg]:
                morphology[seg,1] = 3
            elif 'axon' in segments_dict[seg]:
                morphology[seg,1] = 2
            elif 'basal' in segments_dict[seg]:
                morphology[seg,1] = 4
            else:
                morphology[seg,1] = 5

            morphology[seg,2:5] = coords[seg]
            morphology[seg,5] = segdiam[seg]/2
            morph2[seg,0] = number
            morph2[seg,1:4] = coords[seg]
            morph2[seg,4:] = ends[seg]
        for seg in segments:
            number = seg + 1         
            parent_seg = seg - 1

            if seg == 0:
                morphology[seg,6] = -1
            elif seg < nseg - 1:
                check_parent = np.isclose(coords[seg],ends[parent_seg])
                if check_parent[0] and check_parent[1] and check_parent[2]:
                    morphology[seg,6] =  morphology[parent_seg,0]
                else: #branchpoints
                    changed = 0
                    for i, end in enumerate(ends):
                        check_parent = np.isclose(coords[seg],end)
                        if check_parent[0] and check_parent[1] and check_parent[2]:
                            morphology[seg,6] = morphology[i,0]
                            changed = 1

                            
        morph_path = os.path.join(self.new_path,'morphology')
        if not os.path.exists(morph_path):
            print("Creating",morph_path)
            os.makedirs(morph_path)
        fname = os.path.join(morph_path,self.cell_name)+'.swc'
        print('Saving morphology to',fname)
        np.savetxt(fname, morphology, header='')
        
        
    def add_electrodes(self):
      self.electrode_parameters['x'] =  self.ele_coordinates[:,0],        # x,y,z-coordinates of contact points
      self.electrode_parameters['y'] =  self.ele_coordinates[:,0],        # x,y,z-coordinates of contact points
      self.electrode_parameters['z'] =  self.ele_coordinates[:,0],        # x,y,z-coordinates of contact points
      self.electrode_parameters['sigma'] = self.sigma
      self.simulation_parameters['electrode'] = LFPy.RecExtElectrode(**self.electrode_parameters)
    
    def setup_LFPy_2D_grid(self, eldistribute,orientation,colnb,rownb,xmin,xmax,ymin,ymax,cellelectrodedist,triside,ssNB):
        

        if orientation == 1:
            i, j, k = 1, 2, 0
        if orientation == 2:
            i, j, k  = 2, 0, 1
        if orientation == 3:
            i, j,k = 0, 1, 2
        
        self.ele_coordinates = np.ones((rownb*colnb,3))
    
        if eldistribute == 1: ###grid
            self.ele_coordinates[:,i] = np.array(colnb*list(np.linspace(xmin,xmax,rownb)))
            self.ele_coordinates[:,j] = np.repeat(np.linspace(ymin,ymax,colnb),rownb)
        elif eldistribute == 2:

            self.ele_coordinates[:,i] = np.array(colnb*list(np.linspace(xmin,xmax,rownb)))
            self.ele_coordinates[:,j] = np.repeat(np.linspace(ymin,ymax,colnb),rownb)

        elif eldistribute == 3:
            assert (rownb % 2 == 0), "For hexagon grids row number needs to be even"
            triheight = triside*np.cos(np.pi/6)
            rownb = rownb//2
            triX1 = xmin + triside*np.arange(1,colnb+1) - triside
            triX2 = xmin - triside/2 + triside*np.arange(1,colnb+1)
            triY1 = ymin + 2*triheight*np.arange(1,rownb+1) - triheight
            triY2 = ymin + 2*triheight*np.arange(1,rownb+1)

            grid1 = [(x,y) for x in triX1 for y in triY1]
            grid2 = [(x,y) for x in triX2 for y in triY2]

            Xcoord = grid1[0].extend(grid2[0])
            Ycoord = grid1[1].extend(grid2[1])

            self.ele_coordinates[:,i] = Xcoord
            self.ele_coordinates[:,j] = Ycoord
        
            self.ele_coordinates[:,k] *= cellelectrodedist

        elif eldistribute == 4:
            self.ele_coordinates = np.loadtxt(os.path.join(path,'simulation/ElcoordsDomi14.txt'))
        
        if not os.path.exists(self.new_path):
            print("Creating",self.new_path)
            os.makedirs(self.new_path)
        
            self.add_electrodes()

    def constant_current_injection(self,amp,idx=0):
        self.point_process['idx'] = idx
        self.point_process['amp'] = amp
        self.point_process['dur'] = self.cell.tstopms
        self.point_process['delay'] = 2
        stimulus = LFPy.StimIntElectrode(cell, **self.pointprocess)
        
    def cosine_current_injection(self,amp,idx=0):
        self.point_process['idx'] = idx
        self.point_process['dur'] = 1


        times_stim= np.arange(self.point_process['dur'])
        time_series = np.array(3.6*np.sin(2.*3.141*6.5*times_stim/1000.))
        for istim in range(self.point_process['dur']):
            self.point_process['amp'] = time_series[istim]
            self.point_process['delay'] = istim

            stimulus = LFPy.StimIntElectrode(cell, **self.pointprocess)
            
    def random_synaptic_input(self,lambd=2,tstart=0,tstop=70):
        self.synapse_parameters['idx'] = 0
        pre_syn_sptimes = self.stationary_poisson(self.n_pre_syn, lambd, tstart, tstop)
        self.pre_syn_pick = np.random.permutation(np.arange(self.n_pre_syn))[0:self.n_synapses]

        for i_syn in range(self.n_synapses):
            syn_idx = int(self.cell.get_rand_idx_area_norm())
            self.synapse_parameters.update({'idx' : syn_idx})
            synapse = LFPy.Synapse(self.cell, **self.synapse_parameters)

            synapse.set_spike_times(pre_syn_sptimes[self.pre_syn_pick[i_syn]])
            
    def simulate(self,stimulus=None):
        if stimulus:
            self.stimulus = stimulus
        if self.stimulus == 'cosine':
            self.constant_current_injection(amp=3.6,idx=0)
        elif self.stimulus == 'constant':
            self.constant_current_injection(amp=1,idx=0)
        elif self.stimulus == 'random':
            self.random_synaptic_input()
            
        self.cell.simulate(**self.simulation_parameters)
        
    def save_LFP(self,directory=''):
        LFP_path = os.path.join(self.new_path,directory)
        if not os.path.exists(LFP_path):
            print("Creating",LFP_path)
            os.makedirs(LFP_path)
        fname = os.path.join(LFP_path,'MyLFP')
        np.savetxt( fname, self.simulation_parameters['electrode'].LFP)

        
    def save_electrode_pos(self,directory=''):
        electr = np.hstack((self.ele_coordinates[:,0],self.ele_coordinates[:,1],self.ele_coordinates[:,2]))
        elcoord_x_y_x_path = os.path.join(self.new_path,directory)
        if not os.path.exists(elcoord_x_y_x_path):
            print("Creating",elcoord_x_y_x_path)
            os.makedirs(elcoord_x_y_x_path)
        fname = os.path.join(elcoord_x_y_x_path,'elcoord_x_y_x')
        np.savetxt(fname,electr)
        
    def save_for_R_kernel(self,directory=''):
        self.save_LFP(directory)
        self.save_electrode_pos(directory)
        if directory:
            new_path = directory
        else:
            new_path = self.new_path
        np.savetxt(os.path.join(new_path, 'somav.txt'), self.cell.somav)
        coords = np.hstack((self.cell.xmid, self.cell.ymid, self.cell.zmid) 
        )

        np.savetxt(os.path.join(new_path,  'coordsmid_x_y_z'),coords)
        #coordinates of the segment's beginning
        coordsstart = np.hstack( (self.cell.xstart, self.cell.ystart, self.cell.zstart) )
        
        np.savetxt(os.path.join(new_path,  'coordsstart_x_y_z'),coordsstart)
        
        #coordinates of the segment's end
        coordsend = np.hstack((self.cell.xend, self.cell.yend, self.cell.zend))
        
        np.savetxt( os.path.join(new_path, 'coordsend_x_y_z'),coordsend)
        #sdiameter of the segments
        segdiam = np.hstack((self.cell.diam))
        
        np.savetxt( os.path.join(new_path, 'segdiam_x_y_z'),segdiam)
        
        #length of segments
        np.savetxt( os.path.join(new_path, 'seglength'),self.cell.length)
        #time in the simulation
        np.savetxt( os.path.join(new_path, 'time'),self.cell.tvec)
        
        #lets write to file the simulation locations
        
        np.savetxt( 'synapse_locations',self.pre_syn_pick)

    def save_skCSD_python(self):
        self.save_morphology_to_file()
        self.save_LFP('LFP')
        self.save_electrode_pos('electrode_positions')
    def return_paths_skCSD_python(self):
        return self.new_path
if __name__ == '__main__':
    c = CellModel(morphology=1,cell_name='ball_stick_8',colnb=1,rownb=8,xmin=-500,xmax=500,ymin=-500,ymax=500)
    c.simulate()
    c.save_skCSD_python()
    c.save_for_R_kernel()
#ssNb -- set up seed -- optional arguments
