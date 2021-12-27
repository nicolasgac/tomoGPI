#bash!

if [ -d bin ]
then
echo "bin exist"
else
mkdir bin
fi

if [ -d lib ]
then
echo "lib exist"
else
mkdir lib
fi

if [ -d inc ]
then
echo "inc exist"
else
mkdir inc
fi
