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
	for j in {2..30..2};
	do
		python3 main.py $i segment 1 $j > ./log/resample_lstm/segment-1/$j.$i.log
		# python3 main.py $i segment 0 $j 1 > ./log/non_resample_lstm/segment-1/$j.$i.log
	done
done

# mkdir ./log/resample_lstm/segment-2
# mkdir ./log/non_resample_lstm/segment-2
# for i in {1..10};
# do
# 	for j in {4..20..2};
# 	do
# 		python3 main.py segment 2 $j 0 $i > ./log/resample_lstm/segment-2-0/$j.$i.log
# 	done
# done
#
# mkdir ./log_2/segment-3-3
# for i in {1..10};
# do
# 	for j in {3..15};
# 	do
# 		python3 main.py segment 3 $j 3 $i > ./log_2/segment-3-3/$j.$i.log
# 	done
# done
#
# mkdir ./log_2/segment-4-3
# for i in {1..10};
# do
# 	for j in {3..15};
# 	do
# 		python3 main.py segment 4 $j 3 $i > ./log_2/segment-4-3/$j.$i.log
# 	done
# done
