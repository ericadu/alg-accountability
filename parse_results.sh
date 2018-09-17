for F in exp1a exp1b exp2 exp3 exp4a exp4b exp5a exp5b exp6a exp6b exp7a exp7b; do
  python plot_summary.py --sys1=fairml --sys2=fairtest --exp=$F
done