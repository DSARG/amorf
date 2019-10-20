import plotly.figure_factory as ff 
import plotly as plotly

df = [
      dict(Task="Adding NN Functionality", Start='2019-09-09', Finish='2019-09-24'),
      dict(Task="Parallelizing MLSSVR", Start='2019-09-09', Finish='2019-10-09'), 
      dict(Task="Gathering Literary Sources", Start='2019-10-09', Finish='2019-12-09'),
      dict(Task="GUI, Refactoring, Testing, Deployment", Start='2019-10-09', Finish='2019-11-09'),
      dict(Task="Documentation incl Literature", Start='2019-11-09', Finish='2019-12-09'),
      dict(Task="Analysis of Results + Fine Tuning", Start='2019-11-09', Finish='2020-01-09'),
      dict(Task="Christmas", Start='2019-12-24', Finish='2020-01-01'), 
      dict(Task="Thesis", Start='2020-01-09', Finish='2020-02-24'), 
      dict(Task="Final Review + Printing", Start='2020-02-25', Finish='2020-03-09')]


fig = ff.create_gantt(df,showgrid_x=True, showgrid_y=True, title='Planned Progress')
fig.show() 
plotly.offline.plot(fig,output_type='file')