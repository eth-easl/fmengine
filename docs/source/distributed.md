# Distributed Training

## Multi-GPU training

## Multi-host training

FMEngine support multi-host training with deepspeed. To run multi-host training, you need to install [`pdsh`](https://github.com/chaos/pdsh) first, by running the following command:

```bash
git clone https://github.com/chaos/pdsh.git
cd pdsh
./configure --enable-static-modules --without-rsh --with-ssh --without-ssh-connect-timeout-option --prefix=/your/preferred/path
make
make install
```

## Slurm and HPC