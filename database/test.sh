cat tmp.txt | while read line
do
  mv $line noconverged/
done

