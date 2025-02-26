# Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning


This is the implementation of our paper "[Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning]()" in ICLR 2025.

## Getting Started

**Create Conda Environment**

Install Python environment with conda:
```bash
conda create -n ExpoComm python=3.8
conda activate ExpoComm
```

**Install Epymarl**
```bash
cd src
pip install -r requirements.txt
cd ..
```

**Install IMP environment**

```bash
pip install git+https://github.com/moratodpg/imp_marl.git
cd ..
```

**Install MAgent environment**

```bash
pip install magent==0.1.14
pip install pettingzoo==1.12.0
cp env/battle_v3_view7.py PATH_TO_YOUR_PETTINGZOO_ENV/pettingzoo/magent/
cp env/adversarial_pursuit_view8_v3.py PATH_TO_YOUR_PETTINGZOO_ENV/pettingzoo/magent/
cd ..
```

To ease the environment setup, we also provide the environmental setup we used containing detailed version information in `ExpoComm_env.yaml`. 

## Acknowledgement
The code is implement based on the following open-source projects
- [EPyMARL](https://github.com/uoe-agents/epymarl)

and the following MARL environments:
- [IMP-MARL](https://github.com/moratodpg/imp_marl)
- [MAgent 2](https://github.com/Farama-Foundation/MAgent2)

Please refer to those repo for more documentation.

## Run an experiment 

```shell
python src/main.py --config=[Algorithm name] --env-config=[Env name] --exp-config=[Experiment name]
```

The config files are all located in `src/config`.

`--config` refers to the config files in `src/config/algs`.
`--env-config` refers to the config files in `src/config/envs`.
`--exp-config` refers to the config files in `src/config/exp`. If you want to change the configuration of a particular experiment, you can do so by modifying the yaml file here.

All results will be stored in the `work_dirs` folder.

For example, run ExpoComm on Adversarial Pursuit with 25 predators:

```shell
python src/main.py --config=ExpoComm_one_peer_n6 --env-config=MAgent_AdvPursuit --exp-config=ExpoComm_AdvPursuit45_s0
```


## Citing

If you use this code in your research or find it helpful, please consider citing our paper:
```
@inproceedings{liexponential,
  title={Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning},
  author={Li, Xinran and Wang, Xiaolu and Bai, Chenjia and Zhang, Jun},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025}
}
```