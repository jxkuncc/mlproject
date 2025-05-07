"""
Author: Jorian Khan
Purpose: This code was written for the MEGR Machine Learning in Manufacturing 
        course in the spring of 2025 at University of North Carolina at 
        Charlotte

Date: April 25, 2025
"""
import numpy as np
import json
import pyvista as pv
import matplotlib.colors as mplcolors
from cycler import cycler
from copy import deepcopy
from shapelib.Fits import best_fit_transform, apply_htm

class Shape():
    # TODO: Clean up plotting colors cycler etc
    # TODO: Docstrings and function headers
    # TODO: assertions
    

    def __init__(self, vertices:np.ndarray, variability:list[dict]=None, vertex_labels:list[str] = None):
        self.vertices = vertices
        self.variability = variability
        # if self.variability is None:
        #     self.variability = Shape.normal_variability(self.vertices.shape[0])
        self.vertex_labels = vertex_labels
    @property
    def vertices(self)->np.ndarray:
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices:np.ndarray)->None:
        if vertices.shape[0] < 3:
            raise ValueError('Shapes must consist of at least 3 vertices')
        if vertices.shape[1] !=3: 
            raise ValueError('Shape vertices must be Nx3. Where columns are (x,y,z) coordinates')
        self._vertices = vertices

    @property
    def variability(self)->list:
        return self._variability

    @variability.setter
    def variability(self, variability:list)->None:
        #Example vertex Variability for (3 verts)
        # [
        #     {'name':'uniform', 'shift':0.1, 'params':{'low':-1, 'high':1}}, 
        #     {'name':'uniform', 'shift':0.1, 'params':{'low':-1, 'high':1}}, 
        #     {'name':'uniform', 'shift':0.1, 'params':{'low':-1, 'high':1}}
        # ]
        # if len(variability) != self.vertices.shape[0]: 
        #     raise ValueError('Variability size must match the number of vertices')
        self._variability = variability
    
    @property
    def vertex_labels(self) -> list[str]:
        return self._vertex_labels
    
    @vertex_labels.setter
    def vertex_labels(self, labels:list[str])->None:
        self._vertex_labels = labels

    def save(self, file_path:str):
        #save the same to a json file
        assert self.vertices.shape[0] * 3 == len(self.variability)

        with open(file_path, 'wt') as file:
            data = {}
            for vert_idx in range(self.vertices.shape[0]):
                vertex_info = {
                    'xyz':self.vertices[vert_idx].tolist(),
                    'xyz_distributions': self.variability[vert_idx:vert_idx+3]
                    }

                if self.vertex_labels is not None:
                    data[self.vertex_labels[vert_idx]] = vertex_info
                else:
                    data[f'Vertex_{vert_idx}'] = vertex_info
                    
            
            file.write(json.dumps(data, indent=4))

    def variability_matrix(self)->np.ndarray:
        var_mat = []
        for vert_idx in range(self.vertices.shape[0]):
            # Get the variability for the vertex coordinates
            xyz_variability = self.variability[vert_idx:vert_idx+3]

            vert_xyz_var = []
            for coordinate_variability in xyz_variability:
                dist_name = coordinate_variability['name']
                if dist_name != 'normal': raise NotImplementedError(f'Standard deviation for {dist_name} not implemented')
                vert_xyz_var.append(coordinate_variability['params']['scale'])
            var_mat.append(vert_xyz_var)

        return np.array(var_mat)

    def deformation(self, rng:np.random.Generator = None)->np.ndarray:
        #Given a shape and how each point varies
        # Return a deformation
        assert self.vertices.shape[0] * 3 == len(self.variability)

        if rng is None: rng = np.random.default_rng()

        #Sample by vertex and coordinate 
        deformation = []
        for vert_idx in range(self.vertices.shape[0]):
            # Get the variability for the vertex coordinates
            xyz_variability = self.variability[vert_idx:vert_idx+3]

            #x,y,z sample for the vertex
            vertex_deformation = []
            for coordinate_variability in xyz_variability:
                sampling_func = getattr(rng, coordinate_variability['name'])
                vertex_deformation.append(
                    Shape._sample(sampling_func, coordinate_variability['shift'], **coordinate_variability['params']))

            deformation.append(vertex_deformation)
        
        return np.array(deformation)
    
    def multi_deformations(self, samples:int = 10, rng:np.random.Generator = None):
        #return many deformations
        deformations = []
        for _ in range(samples):
            deformations.append(self.deformation(rng))
        return np.array(deformations)
    
    def deform(self, deformation:np.ndarray=None, inplace:bool=True):
        if deformation is None: deformation = self.deformation()
        assert self.vertices.shape == deformation.shape

        if inplace:
            self.vertices = self.vertices + deformation
        else:
            new_shape = deepcopy(self)
            new_shape.deform(deformation, inplace=True)
            return new_shape

    def best_fit_to(self, another_shape)->np.ndarray:
        #Just for pylance
        another_shape:Shape = another_shape
        #Do the fit
        htm, _, _ = best_fit_transform(self.vertices, another_shape.vertices)
        #Apply the fit to self
        self.vertices = apply_htm(htm, self.vertices)

    def plot_points(self, show_labels:bool= False, **kwargs)->None:
        colors = cycler('color', mplcolors.XKCD_COLORS.values())()   
        plt = pv.Plotter()
        #plot by point        
        for vertex_idx in range(self.vertices.shape[0]):
            plt.add_points(self.vertices[vertex_idx], color =next(colors)['color'],  **kwargs)
        if show_labels:plt.add_point_labels(self.vertices, self.vertex_labels)
        plt.show()

    def plot_cloud(self, samples:int=10, rng:np.random.Generator = None)->None:
        #Apply the deformations to get a cloud
        applied = np.tile(self.vertices, (samples,1,1)) + self.multi_deformations(samples, rng)
        colors = cycler('color', mplcolors.BASE_COLORS.values())()  
        plt = pv.Plotter()
        #plot by point        
        for vertex_idx in range(self.vertices.shape[0]):
            plt.add_points(applied[:,vertex_idx,:], opacity=0.5, color=next(colors)['color'])
        plt.add_points(self.vertices, color = 'r')
        plt.show()

    @classmethod
    def gen_variability(cls, num_verts:int, dist_name:str, shift:float, params:dict)->list[dict]:
        return [{'name':dist_name, 'shift':shift, 'params':params}]* num_verts *3

    @classmethod
    def normal_variability(cls, num_verts:int, shift:float=0, loc:float=0, scale:float=.1):
        return Shape.gen_variability(num_verts, 'normal', shift, params={'loc':loc, 'scale':scale})

    @classmethod
    def uniform_variability(cls, num_verts:int, shift:float=0, low:float=-.1, high:float=.1):
        return Shape.gen_variability(num_verts, 'uniform', shift, params={'low':low, 'high':high})

    @classmethod
    def default_normal(cls, num_verts:int=3, variability:list[dict]=None):
        return Shape(
            np.random.normal(0,1,(num_verts, 3)),
            Shape.normal_variability(num_verts) if variability is None else variability
        )

    @classmethod
    def default_uniform(cls, num_verts:int=3, variability:list[dict]=None):
        return Shape(
            np.random.uniform(-1,1,(num_verts, 3)),
            Shape.uniform_variability(num_verts) if variability is None else variability
        )

    @classmethod
    def from_file(cls, file_path:str):
        vertices = []
        variability = []
        vertex_labels = []

        with open(file_path, 'rt') as file:
            info = json.loads(file.read())
            for vertex_label, vertex_info in info.items(): 
                vertices.append(np.array(vertex_info['xyz']))
                variability.extend(vertex_info['xyz_distributions'])
                vertex_labels.append(vertex_label)
        
        return Shape(np.array(vertices, dtype=np.float32), variability, vertex_labels)

    @classmethod
    def plt_similar(cls, shapes:list, labels:list[str]=None, colors:list=None, show_points:bool=True):
        colors_cycle = cycler('color', mplcolors.XKCD_COLORS.values())() 
        plt = pv.Plotter()

        if labels is None: labels = [None]*len(shapes)
        if colors is None: colors = [None]*len(shapes)

        items = zip(shapes, labels, colors)

        for shape, label, color in items:
            if color is None: color = next(colors_cycle)['color']
            plt.add_points(shape.vertices, label = label, color=color)

        if show_points:
            plt.add_point_labels(shapes[0].vertices, [str(x+1) for x in range(shapes[0].vertices.shape[0])])


        if labels is not None: plt.add_legend()

        plt.show()



    def _sample(func, shift:float=0, **kwargs)->float:      
        return func(**kwargs)+shift
    
    def __add__(self, other):
        if isinstance(other,Shape):
            return self.vertices + other.vertices
        return self.vertices + other
    
    def __sub__(self, other):
        if isinstance(other,Shape):
              return self.vertices - other.vertices
        return self.vertices + other

    def __repr__(self):
        return f'A Shape with {self.vertices.shape[0]} vertices'
    
    def __str__(self):
        return str(self.vertices)