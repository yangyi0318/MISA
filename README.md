# MISA

This is the code for paper "Margin-Guided Clean-Label Backdoor Attacks with Weak Implantation and Strong Activation".

## How to run it

CIFAR-10:

```bash
python MISA.py
--data_root./data
--save_dir
./runs/cifar10
--model_name
resnet18
--dataset
cifar10
--target_label
0
--poison_ratio_target
200
--sur_epochs
100
--trig_steps
2000
--vic_epochs
200
--eps
0.0157
--trigger_amplification
1.0
2.0
3.0
4.0
5.0
--amp
```

CIFAR-100:

```
python MISA.py
--data_root./data
--save_dir
./runs/cifar100
--model_name
resnet18
--dataset
cifar100
--target_label
0
--poison_ratio_target
200
--sur_epochs
200
--trig_steps
2000
--vic_epochs
200
--eps
0.0157
--trigger_amplification
1.0
2.0
3.0
4.0
5.0
--amp
```

The parameter configuration descriptions can be found in MISA.py.

