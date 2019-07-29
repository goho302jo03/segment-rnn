#! /bin/bash
# mkdir ./log_2/full
# for i in {1..10};
# do
# 	python3 main.py full 999 999 999 $i > ./log_2/full/$i.log
# done
#
# mkdir ./log_2/segment-1-0
# for i in {1..10};
# do
# 	for j in {5..25};
# 	do
# 		python3 main.py segment 1 $j 0 $i > ./log_2/segment-1-0/$j.$i.log
# 	done
# done
#
# mkdir ./log_2/segment-2-0
# for i in {1..10};
# do
# 	for j in {5..15};
# 	do
# 		python3 main.py segment 2 $j 0 $i > ./log_2/segment-2-0/$j.$i.log
# 	done
# done
#
# mkdir ./log_2/segment-3-0
for i in {1..10};
do
	for j in {11..15};
	do
		python3 main.py segment 3 $j 0 $i > ./log_2/segment-3-0/$j.$i.log
	done
done

# mkdir ./log_2/segment-4-0
for i in {1..10};
do
	for j in {11..15};
	do
		python3 main.py segment 4 $j 0 $i > ./log_2/segment-4-0/$j.$i.log
	done
done
