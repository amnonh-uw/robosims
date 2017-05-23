if [ "$#" -ne 1 ]; then
    echo "gen_all.bash secondary-storage-dir"
    exit
fi
x=$1

for i in gen_dataset/*.yaml
do
    invoke gen_dataset --config=$" >>& gen_dataset.log
    dataset=`fgrep dataset $i | awk '{ print $2 }'`
    mv $dataset.data $x
    mv $dataset.idx $x
    ln -s $x/$dataset.data .
    ln -s $x/$dataset.idx .
done
