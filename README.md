# RecSys-Training-Planner

## Usage

Clone https://github.com/STAR-Laboratory/Accelerating-RecSys-Training and place files accordingly.

Code base:

```
commit 396409aa1fe3eb606c726bc3f6245b44201f30c8 (origin/main, origin/HEAD, main)
Author: madnan92 <adanan@ece.ubc.ca>
Date:   Sun Sep 17 17:10:02 2023 -0700

    Updated
```

### Necessary code modifications for my pytorch runtime environment

Note: These modifications are specifically for python 3.8.12 + pytorch 1.10. Different software environment might need or need not these modifications to run the stock FAE codes.

1. Replace

```python
with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
```

With

```python
with torch.autograd.profiler.profile(enabled=args.enable_profiling, use_cuda=use_gpu) as prof:
```

2. Replace

dlrm\_fae.py line 1390 and line 1726

```python
hot_row = emb_dict[(emb_no, emb_row)]
```

With

```python
hot_row = int(emb_dict[(emb_no, emb_row)])
```

3. Add

```bash
...... \
--arch-embedding-size="987994-4162024-9439"\
```

to the end of 'TBSM\run\_fae\_profiler.sh'

4. TBSM/tbsm\_fae.py line 714

Replace

```python
hot_row = emb_dict[(emb_no, emb_row)]
```

With

```python
hot_row = int(emb_dict[(emb_no, emb_row)])
```

## Note

Both qr\_flag and md\_flag for the embedding layer are not supported.

num\_workers in dataloader is not supported.
