#! /bin/bash
mkdir ./log/full
for i in {1..10};
do
	python3 main.py $i full > ./log/full/$i.log
done

mkdir -p ./log/resample_lstm/segment-1
mkdir -p  ./log/non_resample_lstm/segment-1
for i in {1..10};
do
	for j in {5..100..5};
	do
		python3 main.py $i segment 1 $j > ./log/resample_lstm/segment-1/$j.$i.log
		python3 main.py $i segment 0 $j 1 > ./log/non_resample_lstm/segment-1/$j.$i.log
	done
done


# mkdir ./log/segment-3-0
# for i in {1..10};
# do
# 	for j in {5..15};
# 	do
# 		python3 main.py segment 3 $j 0 $i > ./log/segment-3-0/$j.$i.log
# 	done
# done
#
# mkdir ./log/segment-4-0
# for i in {1..10};
# do
# 	for j in {5..15};
# 	do
# 		python3 main.py segment 4 $j 0 $i > ./log/segment-4-0/$j.$i.log
# 	done
# done
