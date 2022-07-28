p=$CODE_MEMORY_ERRORS/.pyenvs/merr/lib/python3.9
p64=$CODE_MEMORY_ERRORS/.pyenvs/merr/lib64/python3.9
export PYTHONPATH=$p/site-packages:$PYTHONPATH
export PYTHONPATH=$p64/site-packages:$PYTHONPATH
## put to end
#PYH=/p/software/jusuf/stages/2022/software/Python/3.9.6-GCCcore-11.2.0
#export PYTHONPATH=$PYTHONPATH:$PYH/lib/python3.9/site-packages
#export PYTHONPATH=$PYTHONPATH:$PYH/lib64/python3.9/site-packages
#export LD_LIBRARY_PATH=$p/site-packages:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$p64/site-packages:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib64

cd $CODE_MEMORY_ERRORS
venvname=merr
echo "Activate $venvname venv"
source $CODE_MEMORY_ERRORS/.pyenvs/$venvname/bin/activate
alias pip='python -m pip'
export PATH
