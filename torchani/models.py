# -*- coding: utf-8 -*-
"""The ANI model zoo that stores public ANI models.

Currently the model zoo has three models: ANI-1x, ANI-1ccx, and ANI-2x.
The parameters of these models are stored in `ani-model-zoo`_ repository and
will be automatically downloaded the first time any of these models are
instantiated. The classes of these models are :class:`ANI1x`, :class:`ANI1ccx`,
and :class:`ANI2x` these are subclasses of :class:`torch.nn.Module`.
To use the models just instantiate them and either
directly calculate energies or get an ASE calculator. For example:

.. _ani-model-zoo:
    https://github.com/aiqm/ani-model-zoo

.. code-block:: python

    ani1x = torchani.models.ANI1x()
    # compute energy using ANI-1x model ensemble
    _, energies = ani1x((species, coordinates))
    ani1x.ase()  # get ASE Calculator using this ensemble
    # convert atom species from string to long tensor
    ani1x.species_to_tensor(['C', 'H', 'H', 'H', 'H'])

    model0 = ani1x[0]  # get the first model in the ensemble
    # compute energy using the first model in the ANI-1x model ensemble
    _, energies = model0((species, coordinates))
    model0.ase()  # get ASE Calculator using this model
    # convert atom species from string to long tensor
    model0.species_to_tensor(['C', 'H', 'H', 'H', 'H'])
"""
import os
import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple
from .nn import PairwiseModel, SpeciesConverter, SpeciesEnergies,ANIModel
from .aev import AEVComputer, NbList,SpeciesAEV
from .utils import EnergyShifter,EnergyScaler
from collections import OrderedDict


class SpeciesEnergiesQBC(NamedTuple):
    species: Tensor
    energies: Tensor
    qbcs: Tensor


class BuiltinModel(torch.nn.Module):
    r"""Private template for the builtin ANI models """

    def __init__(self, species_converter, aev_computer, neural_networks, energy_shifter
          , species_to_tensor, consts, sae_dict, periodic_table_index
          , energy_scaler=None):
        super().__init__()
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self._species_to_tensor = species_to_tensor
        self.species = consts.species
        self.periodic_table_index = periodic_table_index
        self.energy_scaler = energy_scaler

        # a bit useless maybe
        self.consts = consts
        self.sae_dict = sae_dict
    
    @classmethod
    def from_dict(cls, model_struct, periodic_table_index=False,**kwargs):
        from . import neurochem
        species=model_struct["species"]
        num_species=len(species)
        consts=neurochem.Constants.from_dict(species,model_struct["aev"])
        aev_computer=AEVComputer(**consts)
        if "aev_normalizer" in model_struct["aev"]:
          aev_computer.set_aev_normalizer(consts.aev_normalizer)
        if "self_energies" in model_struct:
          self_energies=model_struct["self_energies"]
          fit_intercept= len(self_energies)==num_species+1
          if not fit_intercept: assert len(self_energies)==num_species
          energy_shifter = EnergyShifter(self_energies,fit_intercept)
        else:
          energy_shifter = EnergyShifter([0.]*num_species)
        if 'energy_scaler' in model_struct:
          shift,scale = model_struct['energy_scaler']
          energy_scaler=EnergyScaler(shift,scale)
        else:
          energy_scaler=None
        species_converter=SpeciesConverter(species)

        if 'network_type' in model_struct:
            network_type = str(model_struct['network_type']).upper()
        else:
            network_type = 'ANIMODEL'
        
        if network_type == 'ANIMODEL':
          input_size=aev_computer.aev_length
          nn=ANIModel.from_dict(species,model_struct["network_setup"],input_size)
        elif network_type == 'PAIRWISE':
          aev_computer.enable_pairwise_encoding()
          input_sizes=(aev_computer.pairwise_encoding_length
                        ,aev_computer.aev_length
                        ,num_species)
          nn=PairwiseModel.from_dict(model_struct["network_setup"]
                  ,aev_computer.Rcr,*input_sizes)
        else:
          raise ValueError("Unknown network type : {}".format(network_type))
        
        return cls(
          species_converter=species_converter,
          aev_computer=aev_computer,
          neural_networks=nn,
          energy_shifter=energy_shifter,
          species_to_tensor=consts.species_to_tensor,
          consts=consts,
          sae_dict=None,
          periodic_table_index=periodic_table_index,
          energy_scaler=energy_scaler
        )
    
    def to_dict(self,save_weights=True, rmse=None,**kwargs):
        data=OrderedDict()
        if rmse is not None:
          data["rmse"]=rmse
        data["species"]=self.consts.species
        data["self_energies"]=self.energy_shifter.self_energies.tolist()
        if isinstance(self.neural_networks,ANIModel):
          data["network_type"]="ANIMODEL"
        elif isinstance(self.neural_networks,PairwiseModel):
          data["network_type"]="PAIRWISE"
        else:
          raise ValueError("Unknown network type : {}".format(self.neural_networks))
        if self.energy_scaler is not None:
          data["energy_scaler"]=[self.energy_scaler.shift,self.energy_scaler.scale]
        data["aev"]=self.aev_computer.to_dict()
        species=self.consts.species
        data["network_setup"]=self.neural_networks.to_dict(species,save_weights)
        return data
    
    @classmethod
    def from_file(cls, path:str,**kwargs):
        if path.endswith(".yaml") or path.endswith(".yml"):
          return cls.from_yaml(path,**kwargs)
        elif path.endswith(".json"):
          return cls.from_json(path,**kwargs)
        elif path.endswith(".pkl"):
          return cls.from_pickle(path,**kwargs)
        else:
          raise ValueError("Unknown file type")
    
    def to_file(self, path:str,**kwargs):
        if path.endswith(".yaml") or path.endswith(".yml"):
          return self.to_yaml(path,**kwargs)
        elif path.endswith(".json"):
          return self.to_json(path,**kwargs)
        elif path.endswith(".pkl"):
          return self.to_pickle(path,**kwargs)
        else:
          raise ValueError("Unknown file type")
          
    
    @classmethod
    def from_yaml(cls, path,**kwargs):
        import yaml
        with open(path,"r") as f:
            model_struct=yaml.load(f, Loader=yaml.SafeLoader)
        return cls.from_dict(model_struct,**kwargs)

    def to_yaml(self,path,default_flow_style=False,**kwargs):
        import yaml
        data = self.to_dict(**kwargs)
        with open(path,"w") as f:
            yaml.dump(data,f, default_flow_style=default_flow_style)

    @classmethod
    def from_json(cls, path,**kwargs):
        import json
        with open(path,"r") as f:
            model_struct=json.load(f)
        return cls.from_dict(model_struct,**kwargs)

    def to_json(self,path,indent=None,**kwargs):
        import json
        data = self.to_dict(**kwargs)
        with open(path,"w") as f:
            json.dump(data,f,indent=indent)
    
    @classmethod
    def from_pickle(cls, path,**kwargs):
        import pickle
        with open(path,"rb") as f:
            model_struct=pickle.load(f)
        return cls.from_dict(model_struct,**kwargs)

    def to_pickle(self,path,**kwargs):
        import pickle
        data = self.to_dict(**kwargs)
        with open(path,"wb") as f:
            pickle.dump(data,f)
      

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False, model_index=0):
        from . import neurochem  # noqa

        # this is used to load only 1 model (by default model 0)
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)
        if (model_index >= ensemble_size):
            raise ValueError("The ensemble size is only {}, model {} can't be loaded".format(ensemble_size, model_index))

        consts = neurochem.Constants(const_file)
        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        network_dir = os.path.join('{}{}'.format(ensemble_prefix, model_index), 'networks')
        neural_networks = neurochem.load_model(consts.species, network_dir)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None,
                nblist: Optional[NbList] = None,
                shift_energies: bool = True) -> SpeciesEnergies:
        """Calculates predicted properties for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: energies for the given configurations

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)

        # check if unknown species are included
        if species_coordinates[0].ge(self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')

        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc,nblist=nblist)
        species_energies = self.neural_networks(species_aevs)
        
        if self.energy_scaler is not None:
            species_energies = self.energy_scaler(species_energies)
        if shift_energies:
            species_energies = self.energy_shifter(species_energies)
        return species_energies

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None,
                        nblist: Optional[NbList] = None,
                        shift_energies: bool = True) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_atomic_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (C, A) for a BuiltinModel.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc,nblist=nblist)
        atomic_energies = self.neural_networks._atomic_energies(species_aevs)
        
        species = species_aevs[0]
        if self.energy_scaler is not None:
          atomic_energies = self.energy_scaler.scale*atomic_energies + self.energy_scaler.shift
        if shift_energies:
          self_energies = self.energy_shifter.self_energies.clone().to(species.device)
          self_energies = self_energies[species]
          self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
          # shift all atomic energies individually
          assert self_energies.shape == atomic_energies.shape
          atomic_energies += self_energies
        return SpeciesEnergies(species, atomic_energies)
    
    def atomic_environments(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None,
                        nblist: Optional[NbList] = None,
                        *args,**kwargs) -> SpeciesAEV:
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        pairwise_encoding_save = self.aev_computer.compute_pairwise_encoding
        if pairwise_encoding_save:
            self.aev_computer.disable_pairwise_encoding()
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc,nblist=nblist)
        if pairwise_encoding_save:
            self.aev_computer.enable_pairwise_encoding()
        return species_aevs
    
    @torch.jit.export
    def all_output(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None,
                        nblist: Optional[NbList] = None,
                        shift_energies: bool = True) -> SpeciesEnergies:
        assert (hasattr(self.neural_networks, '_all_output')
           and callable(getattr(self.neural_networks, '_all_output')))

        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc,nblist=nblist)
        output = self.neural_networks._all_output(species_aevs)

        species = species_aevs[0]
        if 'atomic_energies' in output:
          if self.energy_scaler is not None:
            output["atomic_energies"] = self.energy_scaler.scale*output["atomic_energies"] + self.energy_scaler.shift
          if shift_energies:
            self_energies = self.energy_shifter.self_energies.clone().to(species.device)
            self_energies = self_energies[species]
            self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
            # shift all atomic energies individually
            assert self_energies.shape == output["atomic_energies"].shape
            output["atomic_energies"] += self_energies
          output["energy"] = output["atomic_energies"].sum(dim=1)

        return output

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)

    def species_to_tensor(self, *args, **kwargs):
        """Convert species from strings to tensor.

        See also :method:`torchani.neurochem.Constant.species_to_tensor`

        Arguments:
            species (:class:`str`): A string of chemical symbols

        Returns:
            tensor (:class:`torch.Tensor`): A 1D tensor of integers
        """
        # The only difference between this and the "raw" private version
        # _species_to_tensor is that this sends the final tensor to the model
        # device
        return self._species_to_tensor(*args, **kwargs) \
            .to(self.aev_computer.ShfR.device)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.species, self, **kwargs)


class BuiltinEnsemble(BuiltinModel):
    """Private template for the builtin ANI ensemble models.

    ANI ensemble models form the ANI models zoo are instances of this class.
    This class is a torch module that sequentially calculates
    AEVs, then energies from a torchani.Ensemble and then uses EnergyShifter
    to shift those energies. It is essentially a sequential

    'AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=False), or a sequential

    'SpeciesConverter -> AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=True).

    .. note::
        This class is for internal use only, avoid relying on anything from it
        except the public methods, always use ANI1x, ANI1ccx, etc to instance
        the models.
        Also, don't confuse this class with torchani.Ensemble, which is only a
        container for many ANIModel instances and shouldn't be used directly
        for calculations.

    Attributes:
        species_converter (:class:`torchani.nn.SpeciesConverter`): Converts periodic table index to
            internal indices. Only present if periodic_table_index is `True`.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer with
            builtin constants
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter with
            builtin Self Atomic Energies.
        periodic_table_index (bool): Whether to use element number in periodic table
            to index species. If set to `False`, then indices must be `0, 1, 2, ..., N - 1`
            where `N` is the number of parametrized species.
    """

    def __init__(self, species_converter, aev_computer, neural_networks,
                 energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index
                 ,energy_scaler=None):
        super().__init__(species_converter, aev_computer, neural_networks,
                         energy_shifter, species_to_tensor, consts, sae_dict,
                         periodic_table_index, energy_scaler)

    @torch.jit.export
    def atomic_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                        cell: Optional[Tensor] = None,
                        pbc: Optional[Tensor] = None,
                        nblist: Optional[NbList] = None,
                        shift_energies: bool = True,
                        average: bool = True) -> SpeciesEnergies:
        """Calculates predicted atomic energies of all atoms in a molecule

        see `:method:torchani.BuiltinModel.atomic_energies`

        If average is True (the default) it returns the average over all models
        (shape (C, A)), otherwise it returns one atomic energy per model (shape
        (M, C, A))
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc,nblist=nblist)
        members_list = []
        for nnp in self.neural_networks:
            members_list.append(nnp._atomic_energies((species, aevs)).unsqueeze(0))
        member_atomic_energies = torch.cat(members_list, dim=0)

        if self.energy_scaler is not None:
          member_atomic_energies = self.energy_scaler.scale*member_atomic_energies + self.energy_scaler.shift
        if shift_energies:
          self_energies = self.energy_shifter.self_energies.clone().to(species.device)
          self_energies = self_energies[species]
          self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
          # shift all atomic energies individually
          assert self_energies.shape == member_atomic_energies.shape[1:]
          member_atomic_energies += self_energies
        if average:
            return SpeciesEnergies(species, member_atomic_energies.mean(dim=0))
        return SpeciesEnergies(species, member_atomic_energies)

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False):
        from . import neurochem  # noqa
        # this is used to load only 1 model (by default model 0)
        const_file, sae_file, ensemble_prefix, ensemble_size = neurochem.parse_neurochem_resources(info_file_path)

        consts = neurochem.Constants(const_file)
        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor
        neural_networks = neurochem.load_model_ensemble(consts.species,
                                                        ensemble_prefix, ensemble_size)

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

    def __getitem__(self, index):
        """Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model

        Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model
        or
        Indexing allows access to a single model inside the ensemble
        that can be used directly for calculations. The model consists
        of a sequence AEVComputer -> ANIModel -> EnergyShifter
        and can return an ase calculator and convert species to tensor.

        Args:
            index (:class:`int`): Index of the model

        Returns:
            ret: (:class:`torchani.models.BuiltinModel`) Model ready for
                calculations
        """
        ret = BuiltinModel(self.species_converter, self.aev_computer,
                           self.neural_networks[index], self.energy_shifter,
                           self._species_to_tensor, self.consts, self.sae_dict,
                           self.periodic_table_index)
        return ret

    @torch.jit.export
    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                         cell: Optional[Tensor] = None,
                         pbc: Optional[Tensor] = None,
                         nblist: Optional[NbList] = None,
                         shift_energies: bool = True) -> SpeciesEnergies:
        """Calculates predicted energies of all member modules

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (M, C), where M is the number
                of modules in the ensemble

        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species, aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc,nblist=nblist)
        member_outputs = []
        for nnp in self.neural_networks:
            unshifted_energies = nnp((species, aevs)).energies
            if self.energy_scaler is not None:
                unshifted_energies = self.energy_scaler.scale*unshifted_energies + self.energy_scaler.shift
            shifted_energies = ( self.energy_shifter((species, unshifted_energies)).energies 
                                  if shift_energies else unshifted_energies )
            member_outputs.append(shifted_energies.unsqueeze(0))
        return SpeciesEnergies(species, torch.cat(member_outputs, dim=0))

    @torch.jit.export
    def energies_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                      cell: Optional[Tensor] = None,
                      pbc: Optional[Tensor] = None,
                      nblist: Optional[NbList] = None,
                      unbiased: bool = True,
                      shift_energies: bool = True) -> SpeciesEnergiesQBC:
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            and qbc factors will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not
                enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled
            unbiased: if `True` then Bessel's correction is applied to the
                standard deviation over the ensemble member's. If `False` Bessel's
                correction is not applied, True by default.

        Returns:
            species_energies_qbcs: species, energies and qbc factors for the
                given configurations note that the shape of species is (C, A),
                where C is the number of configurations and A the number of
                atoms, the shape of energies is (C,) and the shape of qbc
                factors is also (C,).
        """
        species, energies = self.members_energies(species_coordinates, cell, pbc, nblist
                                                    ,shift_energies=shift_energies)

        # standard deviation is taken across ensemble members
        qbc_factors = energies.std(0, unbiased=unbiased)

        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        energies = energies.mean(dim=0)
        assert qbc_factors.shape == energies.shape
        return SpeciesEnergiesQBC(species, energies, qbc_factors)

    def __len__(self):
        """Get the number of networks in the ensemble

        Returns:
            length (:class:`int`): Number of networks in the ensemble
        """
        return len(self.neural_networks)
    
    def to_json(self, json_file_path, save_weights=True, indent=None,model_index=None,**kwargs):
        if model_index is not None:
          self[model_index].to_json(json_file_path, save_weights, indent,**kwargs)
        else:
          for i in range(len(self)):
            self[i].to_json(json_file_path.replace(".json",f".{i}.json"), save_weights, indent)


def ANI1x(periodic_table_index=False, model_index=None):
    """The ANI-1x model as in `ani-1x_8x on GitHub`_ and `Active Learning Paper`_.

    The ANI-1x model is an ensemble of 8 networks that was trained using
    active learning on the ANI-1x dataset, the target level of theory is
    wB97X/6-31G(d). It predicts energies on HCNO elements exclusively, it
    shouldn't be used with other atom types.

    .. _ani-1x_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1x_8x

    .. _Active Learning Paper:
        https://aip.scitation.org/doi/abs/10.1063/1.5023802
    """
    info_file = 'ani-1x_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)


def ANI1ccx(periodic_table_index=False, model_index=None):
    """The ANI-1ccx model as in `ani-1ccx_8x on GitHub`_ and `Transfer Learning Paper`_.

    The ANI-1ccx model is an ensemble of 8 networks that was trained
    on the ANI-1ccx dataset, using transfer learning. The target accuracy
    is CCSD(T)*/CBS (CCSD(T) using the DPLNO-CCSD(T) method). It predicts
    energies on HCNO elements exclusively, it shouldn't be used with other
    atom types.

    .. _ani-1ccx_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1ccx_8x

    .. _Transfer Learning Paper:
        https://doi.org/10.26434/chemrxiv.6744440.v1
    """
    info_file = 'ani-1ccx_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)


def ANI2x(periodic_table_index=False, model_index=None):
    """The ANI-2x model as in `ANI2x Paper`_ and `ANI2x Results on GitHub`_.

    The ANI-2x model is an ensemble of 8 networks that was trained on the
    ANI-2x dataset. The target level of theory is wB97X/6-31G(d). It predicts
    energies on HCNOFSCl elements exclusively it shouldn't be used with other
    atom types.

    .. _ANI2x Results on GitHub:
        https://github.com/cdever01/ani-2x_results

    .. _ANI2x Paper:
        https://doi.org/10.26434/chemrxiv.11819268.v1
    """
    info_file = 'ani-2x_8x.info'
    if model_index is None:
        return BuiltinEnsemble._from_neurochem_resources(info_file, periodic_table_index)
    return BuiltinModel._from_neurochem_resources(info_file, periodic_table_index, model_index)
