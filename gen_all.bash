if [ "$#" -eq 2 ]; then
    x=$1
    i=$2
    dataset=`fgrep dataset $i | sed -e 's/dataset: //' | sed -e 's/datasets\///'`
    rm -f datasets/$dataset.data
    rm -f datasets/$dataset.idx
    rm -f $x/$dataset.data
    rm -f $x/$dataset.idx
    touch $x/$dataset.data
    touch $x/$dataset.idx
    ln -s $x/$dataset.data datasets/$dataset.data
    ln -s $x/$dataset.idx datasets/$dataset.idx
    invoke gen_train_dataset --config=$i &> datasets/gen_$dataset.out
    exit
fi

if [ "$#" -ne 1 ]; then
    echo "gen_all.bash secondary-storage-dir"
    exit
fi
x=$1

for i in gen_dataset/*.yaml
do
    echo Generating $i in the background
    bash gen_all.bash $x $i &
    sleep 3
done
