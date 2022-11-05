today=$(date +"%y%m%d")
echo ${today}

mkdir -p ../arc/${today}
cp ../log/* ../arc/${today}/
cp ../share/* ../arc/${today}/
cp ../cfg/* ../arc/${today}/
cp ../lib/IoqHost.py ../arc/${today}/
cp report*.txt ../arc/${today}/

rm ../arc/${today}/ioq.err
rm ../arc/${today}/ioq.out

rm ../log/*

