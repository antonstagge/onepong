import numpy
import data_set as data
import mlp


inputs = []
targets = []
for x in data.data_set:
    inputs.append(list(x.data))
    targets.append([x.good_day])


targets = numpy.array(targets)

p = mlp.mlp(inputs, targets, 10)

for i in range(100):
    p.mlptrain(inputs, targets, 0.01, 20)


user_data = []
print "Rate the following from a scale 1 to 6, 1 being the worst and 6 the best!"
for field in data.field_names:
    print "Rate your %s\n" % field
    rating_input = raw_input("> ")
    try:
        rating = int(rating_input)
        if rating < 1 or rating > 6:
            raise ValueError("Wrong input")
        user_data.append(rating)
    except:
        print "Not a valid input!"
        exit(1)

print "\n"
#user_in = numpy.tile(user_data, (len(inputs), 1))
#user_in = numpy.concatenate((user_in,-numpy.ones((len(inputs),1))),axis=1)
#print(user_in.shape)
user_data.append(1)
user_in = numpy.array(user_data)
user_in = user_in.reshape((1,-1))
print(user_in.shape)
output = p.mlpfwd(user_in)
print "The ANN determined you day as: %f" % output[0][0]
