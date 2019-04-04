#!/usr/bin/env bash
output=`./venv/bin/jupytext --from ipynb --check flake8 $1 2>&1 >/dev/null`
ret=$?
if [ $ret -eq 0 ]; then
    exit 0
fi
echo $1
output=`echo $output | sed -e "s/Command 'flake8 -' exited with code 1: //g"`
python -c "print($output.decode('utf-8'))"
exit $ret
