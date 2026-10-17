pwd
ls
ls -la
mkdir -p week02_system/day01/input   
cd week02_system/day01
printf 'INFO training started\nWARNING slow batch\nERROR out of memory\nERROR timeout\n' > input/train.log
find . -type f
cat input/train.log
wc -l input/train.log
grep 'ERROR' input/train.log
cp input/train.log output/train-copy.log
head -n 2 input/train.log
tail -n 2 input/train.log
