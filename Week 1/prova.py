import graphlab

from graphlab import SFrame

df = SFrame("people-example.csv")

graphlab.canvas.set_target('ipynb')

df['age'].show(view='Categorical')

print "countries " ,df['Country']

print "ages " , df['age']


print "ages max" , df['age'].max()


print "ages mean" , df['age'].mean()