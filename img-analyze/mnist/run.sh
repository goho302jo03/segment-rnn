#! /bin/bash
neuron="lstm"
mkdir ./log/$neuron
mkdir ./log/$neuron/full
for i in {1..3};
do
	python3 main.py $i full > ./log/$neuron/full/$i.log
done

mkdir -p ./log/$neuron/resample/segment-1
for i in {1..3};
do
	for j in {20..60..5};
	do
		python3 main.py $i segment 1 $j   > ./log/$neuron/resample/segment-1/$j.$i.log
	done
done
