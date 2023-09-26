# FasterTransformer for SaumsungCEChallenge

Check out FasterTransformer [README.md](FasterTransformerReadME.md)

## Installation


```
mkdir -p FasterTransformer/build
cd FasterTransformer/build
git submodule init && git submodule update
cmake -DSM=70 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j32
```

