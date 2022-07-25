import torch
from collections import OrderedDict
from torch import Tensor
from typing import Tuple, NamedTuple, Optional, List
from . import utils
from .aev import NbList,SpeciesAEV, SpeciesAEVPairwise


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor


class ANIModel(torch.nn.ModuleDict):
    """ANI model that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`torchani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """

    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super().__init__(self.ensureOrderedDict(modules))

    def forward(self, species_aev:SpeciesAEV,  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None,
                nblist: Optional[NbList] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        atomic_energies = self._atomic_energies((species, aev))
        # shape of atomic energies is (C, A)
        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))

    @torch.jit.export
    def _atomic_energies(self, species_aev:SpeciesAEV) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.values()):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return output
    
    @classmethod
    def from_dict(cls,species,dict,input_size):
        atomic_nets = OrderedDict()
        for atom_type in species:
          nn_struct=dict[atom_type]
          key=nn_struct["key"] if "key" in nn_struct else atom_type
          layers=nn_struct["layers"]
          atomic_nets[key] = SequentialBuilder.from_dicts(input_size,layers)
        return cls(atomic_nets)
    
    def to_dict(self, species, save_weights=True):
        from . import neurochem
        dict=OrderedDict()
        for i,key in enumerate(self.keys()):
          k=species[i]
          dict[k]=OrderedDict()
          if key != k: dict[k]["key"]=key
          dict[k]["layers"]=SequentialBuilder.to_dicts(self[key],save_weights)
        return dict    



class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        super().__init__(modules)
        self.size = len(modules)

    def forward(self, species_input: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None,
                nblist: Optional[NbList] = None) -> SpeciesEnergies:
        sum_ = 0
        for x in self:
            sum_ += x(species_input)[1]
        species, _ = species_input
        return SpeciesEnergies(species, sum_ / self.size)


class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, input_: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None,
                nblist: Optional[NbList] = None):
        for module in self:
            input_ = module(input_, cell=cell, pbc=pbc, nblist=nblist)
        return input_
    
    def append(self,module):
      self.add_module(str(len(self)), module)
      return self

class SequentialBuilder:

    @staticmethod
    def get_out_features(modules):
      out_features=None
      for mod in modules:
        if hasattr(mod,"out_features"):
          out_features=mod.out_features
      return out_features
        
    @staticmethod
    def from_dicts(input_size:int,layers:list):
        i=input_size
        modules=[]
        for layer in layers:
          #print(layer)
          i,module,activation=SequentialBuilder.build_layer(i,layer)
          modules.append(module)
          if activation is not None:
            modules.append(activation)
        return torch.nn.Sequential(*modules)
    
    @staticmethod
    def build_layer(i:int,layer:dict):
        o:int = layer['nodes'] if 'nodes' in layer else i
        type=str(layer['type']).upper()
        if type == 'LINEAR':
          module = torch.nn.Linear(i, o)
          if 'weights' in layer:
            module.weight.data = torch.tensor(layer['weights'])
          if 'bias' in layer: 
            module.bias.data = torch.tensor(layer['bias'])
        elif type == 'DROPOUT': 
          dropout=0.25
          if 'fraction' in layer: dropout=layer['fraction']
          module=torch.nn.Dropout(dropout)
        elif type == 'ALPHADROPOUT':
          dropout=0.25
          if 'fraction' in layer: dropout=layer['fraction']
          module=torch.nn.AlphaDropout(dropout)
        elif type == 'ENHANCE':
          module=NonlinearEnhance(i)
          o=module.out_features
        elif type == 'HARMONIC':
          module=HarmonicModule(i)
          if 'weights' in layer:
            module.weight.data = torch.tensor(layer['weights'])
          if 'bias' in layer: 
            module.bias.data = torch.tensor(layer['bias'])
          if 'shift' in layer: 
            module.shift.data = torch.tensor(layer['shift'])
          o=1
        else:
          raise ValueError("Unsupported layer type: "+type)

        activation = None
        if 'activation' in layer:
          activation = SequentialBuilder.activation_from_name(layer['activation'])

        return o,module,activation

    @staticmethod
    def to_dicts(modules,save_weights=True):
        dicts=[]
        for layer in modules:
          if isinstance(layer,torch.nn.Linear):
            dicts.append({
              "type":'linear',
              "nodes":layer.out_features,
            })
            if save_weights:
              dicts[-1]["weights"]=layer.weight.tolist()
              if layer.bias is not None:
                dicts[-1]["bias"]=layer.bias.tolist()
          elif isinstance(layer,torch.nn.Dropout):
            dicts.append({
              "type":'dropout',
              "fraction":layer.p,
            })
          elif isinstance(layer,torch.nn.AlphaDropout):
            dicts.append({
              "type":'alphadropout',
              "fraction":layer.p,
            })
          elif isinstance(layer,NonlinearEnhance):
            dicts.append({
              "type":'enhance',
            })
          elif isinstance(layer,HarmonicModule):
            dicts.append({
              "type":'harmonic',
            })
            if save_weights:
              dicts[-1]["weights"]=layer.weight.tolist()
              dicts[-1]["shift"]=layer.shift.tolist()
              if layer.bias is not None:
                dicts[-1]["bias"]=layer.bias.tolist()
          else:
            dicts[-1]["activation"]=SequentialBuilder.name_from_activation(layer)
        return dicts
      
    @staticmethod
    def activation_from_name(activation_name):
        activ = str(activation_name).upper()
        if activ == "NONE":
            return None
        if activ == "IDENTITY":
            return torch.nn.Identity()
        elif activ == "GAUSSIAN":  # Gaussian
            return Gaussian()
        elif activ == "CELU":  # CELU
            return torch.nn.CELU(alpha=0.1)
        elif activ == "SELU":  # CELU
            return torch.nn.SELU()
        else:
            raise NotImplementedError(
                'Unexpected activation {}'.format(activ))

    @staticmethod
    def name_from_activation(activation):
        if activation is None:
            return "NONE"
        elif isinstance(activation, torch.nn.Identity):
            return "IDENTITY"
        elif isinstance(activation, Gaussian):
            return "GAUSSIAN"
        elif isinstance(activation, torch.nn.CELU):
            return "CELU"
        elif isinstance(activation, torch.nn.SELU):
            return "SELU"
        else:
            raise NotImplementedError(
                f'Unexpected activation {activation}')
      


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- x * x)


class SpeciesConverter(torch.nn.Module):
    """Converts tensors with species labeled as atomic numbers into tensors
    labeled with internal torchani indices according to a custom ordering
    scheme. It takes a custom species ordering as initialization parameter. If
    the class is initialized with ['H', 'C', 'N', 'O'] for example, it will
    convert a tensor [1, 1, 6, 7, 1, 8] into a tensor [0, 0, 1, 2, 0, 3]

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """
    conv_tensor: Tensor

    def __init__(self, species):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(utils.PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.register_buffer('conv_tensor', torch.full((maxidx + 2,), -1, dtype=torch.long))
        for i, s in enumerate(species):
            self.conv_tensor[rev_idx[s]] = i

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None,
                nblist: Optional[NbList] = None):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        species, coordinates = input_
        converted_species = self.conv_tensor[species]

        # check if unknown species are included
        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f'Unknown species found in {species}')

        return SpeciesCoordinates(converted_species.to(species.device), coordinates)

class HarmonicModule(torch.nn.Module):
    """Harmonic potential module
    out = X^T (W^T W) X + b
    """
    __constants__ = ['in_features']
    shift:Tensor
    weight:Tensor

    def __init__(self, in_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
      factory_kwargs = {'device': device, 'dtype': dtype}
      super(HarmonicModule, self).__init__()
      self.in_features = in_features
      #self.weight = torch.nn.Parameter(torch.empty((in_features, in_features), **factory_kwargs))
      self.weight = torch.nn.Parameter(torch.empty((in_features,in_features), **factory_kwargs))
      self.shift = torch.nn.Parameter(torch.empty(in_features, **factory_kwargs))
      if bias:
          self.bias = torch.nn.Parameter(torch.empty(1, **factory_kwargs))
      else:
          self.register_parameter('bias', None)
      self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        #torch.nn.init.normal_(self.weight, std=1./self.in_features)
        torch.nn.init.eye_(self.weight)/self.in_features
        #torch.nn.init.uniform_(self.weight, b=1./self.in_feature)
        torch.nn.init.zeros_(self.shift)
        if self.bias is not None:
          torch.nn.init.zeros_(self.bias)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, 1, self.bias is not None
        )

    def forward(self, input: Tensor):
        out = torch.sum( 
              ((input - self.shift[None,:]) @ self.weight.t())**2
                , dim=1, keepdim=True)
        #out = torch.sum( 
        #      ((input - self.shift[None,:]) * self.weight[None,:])**2
        #        , dim=1, keepdim=True)
        return out + self.bias[None,:] if self.bias is not None else out

class NonlinearEnhance(torch.nn.Module):
    __constants__ = ['in_features','out_features','eps']

    def __init__(self, in_features: int) -> None:
      super(NonlinearEnhance, self).__init__()
      self.in_features = in_features
      self.out_features = 4*in_features
      self.eps = 1e-3
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

    def forward(self, input: Tensor):
        out= torch.cat((input, input**2,input**3,input**4), dim=1)
        return out

class AllegroMiniLayer(torch.nn.Module):
    accepted_oper=["sum","tensor"]

    def __init__(self, embedding:torch.nn.Module, latent:torch.nn.Module
      , compress:Optional[torch.nn.Module]=None,oper="sum"):
      super().__init__()
      self.embedding=embedding
      self.latent=latent
      self.compress=compress

      oper_ = str(oper).lower()
      assert oper_ in self.accepted_oper
      self.oper = oper_
    
    def forward(self, input:Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor,Tensor,Tensor]:
      x,v,aev = input
      weights = self.embedding(x)
      if self.oper == "sum":
        vo = weights*aev + v
      elif self.oper == "tensor":
        vo = torch.einsum('bi,bj->bij',(weights*aev, v)).view(v.shape[0],aev.shape[1]*v.shape[1])
        if self.compress is not None:
            vo = self.compress(vo)
      else:
        vo = weights*aev
      xo = self.latent(torch.column_stack([x,vo]))
      return xo+x, vo, aev
    
    @classmethod
    def from_dict(cls, d, len_x, len_v, len_aev):
      embedding = SequentialBuilder.from_dicts(len_x,d['embedding'])
      embedding_length = SequentialBuilder.get_out_features(embedding)
      if embedding_length != len_v:
          embedding.add_module(str(len(embedding)),torch.nn.Linear(embedding_length,len_aev))

      oper = str(d['oper']).lower() if 'oper' in d else 'sum'
      compress = None
      if oper == 'tensor':
        if 'compress' in d:
            compress = SequentialBuilder.from_dicts(len_v,d['compress'])
        elif len_v==1:
            compress = None
        else:
            compress = torch.nn.Sequential(torch.nn.Linear(len_v*len_aev,len_aev))
      
      
      latent = SequentialBuilder.from_dicts(len_x+len_aev,d['latent'])
      latent_length = SequentialBuilder.get_out_features(latent)
      if latent_length != len_x:
          latent.add_module(str(len(latent)),torch.nn.Linear(latent_length,len_x))

      return cls(embedding,latent,compress,oper)
    
    def to_dict(self,species,save_weights=True):
      d={}
      d["embedding"] = SequentialBuilder.to_dicts(self.embedding,save_weights)
      d["latent"] = SequentialBuilder.to_dicts(self.latent,save_weights)
      if self.compress is not None:
        d["compress"] = SequentialBuilder.to_dicts(self.compress,save_weights)
      d["oper"] = self.oper
      return d

class AllegroMini(torch.nn.Sequential):
    __constants__ = ['n_layers','out_features']
    def __init__(self, out_features,*modules,oper="sum"):
        super().__init__(*modules)
        self.n_layers=len(modules)
        self.out_features=out_features

        oper_ = str(oper).lower()
        assert oper_ in AllegroMiniLayer.accepted_oper
        self.oper = oper_
    
    def forward(self, input:Tuple[Tensor, Tensor]) -> Tuple[Tensor,Tensor]:
        x, aev = input
        xo = x.clone()
        if self.oper == "sum":
          vo = aev.new_zeros(aev.shape)
        elif self.oper == "tensor":
          vo = aev.new_ones((aev.shape[0],1))
        else:
          raise ValueError("Unknown oper: {}".format(self.oper))
        
        for m in self:
            xo, vo, aev = m((xo,vo,aev))
        return xo
    
    @classmethod
    def from_dict(cls, d, len_x, len_aev):
        n_layers = d['n_layers']
        assert n_layers > 0
        oper = str(d['oper']).lower() if 'oper' in d else 'sum'
        layers=[]
        if 'layers' in d:
          dl = d['layers']
          assert isinstance(dl,list)
          assert len(dl)==n_layers
          layers.append(AllegroMiniLayer.from_dict(dl[0], len_x, 1,len_aev))
          if len(dl)>1:
            for i in range(1,len(dl)):
              layers.append(AllegroMiniLayer.from_dict(dl[i], len_x, len_aev,len_aev))
        else:
          layers.append(AllegroMiniLayer.from_dict(d, len_x, 1, len_aev))
          if n_layers>1:
            for i in range(1,n_layers):
              layers.append(AllegroMiniLayer.from_dict(d, len_x, len_aev, len_aev))
        
        o= cls(len_x,*layers,oper=oper)
        for l in layers:
          assert l.oper == o.oper
        return o
    
    def to_dict(self,species,save_weights=True):
        d={"type":"allegromini","n_layers":self.n_layers,"oper":self.oper}
        d["layers"] = []
        for m in self:
          d["layers"].append(m.to_dict(species,save_weights))
        return d

class PairwiseTransform(torch.nn.Module):
    __constants__ = ['out_features']
    def __init__(self, out_features, embedding=None, latent=None):
        super().__init__()
        self.out_features = out_features
        assert (embedding is not None) or (latent is not None)
        self.embedding = None
        if embedding is not None:
          if isinstance(embedding, torch.nn.Sequential):
            self.embedding = embedding
          else:
            self.embedding = torch.nn.Sequential(embedding)
        
        self.latent = None
        if latent is not None:
          if isinstance(latent, torch.nn.Sequential):
            self.latent = latent
          else:
            self.latent = torch.nn.Sequential(latent)
    
    def forward(self, input:Tuple[Tensor,Tensor]) -> Tensor:
        x, aev = input
        if self.embedding is not None:
          weights = self.embedding(x)
          aev *= weights
        
        if self.latent is not None:
          xo = x + self.latent(torch.column_stack([x,aev]))
        else:
          xo = torch.column_stack([x,aev])
        
        return xo

    @classmethod
    def from_dict(cls,d,input_length,aev_length):
        embedding=None
        latent=None
        if 'embedding' in d:
            embedding=SequentialBuilder.from_dicts(input_length,d["embedding"])
            embedding_length = SequentialBuilder.get_out_features(embedding)
            if embedding_length != aev_length:
                embedding.add_module(str(len(embedding)),torch.nn.Linear(embedding_length,aev_length))
        
        if 'latent' in d:
            latent=SequentialBuilder.from_dicts(input_length+aev_length,d["latent"])
            latent_length = SequentialBuilder.get_out_features(latent)
            if latent_length != input_length:
                latent.add_module(str(len(latent)),torch.nn.Linear(latent_length,input_length))
            out_features = input_length
        else:
            out_features = input_length + aev_length
        return cls(out_features,embedding, latent)
    
    def to_dict(self,species,save_weights=True):
        d={"type":"default"}
        if self.embedding is not None:
            d["embedding"]=SequentialBuilder.to_dicts(self.embedding,save_weights)
        if self.latent is not None:
            d["latent"]=SequentialBuilder.to_dicts(self.latent,save_weights)
        return d
        
class PairwiseModel(torch.nn.Module):
    def __init__(self, modules,embedding=None,transform=None):
      super().__init__()
      if isinstance(modules, torch.nn.Sequential):
        self.energy_model = modules
      else:
        self.energy_model = torch.nn.Sequential(modules)
      
      self.embedding = None
      if embedding is not None:
        if isinstance(embedding, torch.nn.Sequential):
          self.embedding = embedding
        else:
          self.embedding = torch.nn.Sequential(embedding)
      
      self.transform = transform

    def forward(self, input: SpeciesAEVPairwise) -> SpeciesEnergies:
        species=input[0]
        atomic_energies = self._atomic_energies(input)
        # shape of atomic energies is (C, A)
        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))

    
    @torch.jit.export
    def _atomic_energies(self, input: SpeciesAEVPairwise) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        species,aev,central_atom,pairwise_encoding = input
        species_ = species.flatten()

        pairwise_aev=aev.view(aev.shape[0]*aev.shape[1],aev.shape[2])[central_atom,:]

        if self.embedding is not None:
          embedding = self.embedding(pairwise_encoding)
        else:
          embedding = pairwise_encoding
        
        if self.transform is not None:
          pairwise_input = self.transform((embedding,pairwise_aev))
        else:
          pairwise_input = torch.column_stack([embedding,pairwise_aev])
          
        pairwise_output = self.energy_model(pairwise_input)
        output = pairwise_output.new_zeros(species_.shape[0],1)
        output.index_add_(0, central_atom, pairwise_output)
        output= output.view_as(species)
        return output
    
    @classmethod
    def from_dict(cls,d,encoding_length,aev_length):
        embedding=None
        transform=None
        embedding_length=encoding_length
        if 'embedding' in d:
            embedding=SequentialBuilder.from_dicts(encoding_length,d["embedding"])
            embedding_length = SequentialBuilder.get_out_features(embedding)
        
        input_size = embedding_length + aev_length

        if 'transform' in d:
            dt = d["transform"]
            ttype = str(dt["type"]).lower()
            if ttype == "default":
                transform = PairwiseTransform.from_dict(dt,embedding_length,aev_length)
            elif ttype == "allegromini":
                transform = AllegroMini.from_dict(dt,embedding_length,aev_length)
            else:
                raise ValueError("Unknown transform type: {}".format(ttype))
            input_size = transform.out_features

        layers=SequentialBuilder.from_dicts(input_size,d["energy"])
        return cls(layers,embedding,transform)
        
    
    def to_dict(self,species,save_weights=True):
        d={"energy":SequentialBuilder.to_dicts(self.energy_model,save_weights)}
        if self.embedding is not None:
            d["embedding"]=SequentialBuilder.to_dicts(self.embedding,save_weights)
        if self.transform is not None:
            d["transform"]=self.transform.to_dict(species,save_weights)
        return d
      
    