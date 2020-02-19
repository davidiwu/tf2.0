'''
tf.function:
Compiles a function into a callable TensorFlow graph.

tf.function constructs a callable that executes a TensorFlow graph (tf.Graph) 
created by trace-compiling the TensorFlow operations in func, 
effectively executing func as a TensorFlow graph.

'''
import tensorflow as tf

a = tf.Variable(2.0)
b = tf.Variable(3.0)

# use tf.function as decorator:
@tf.function
def add(x, y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b

sth = add(4, 6)
print(sth)
print(6 * 3 + 3 + 6 * 3 * 4)

concrete_func = add.get_concrete_function(4, 6)
print(concrete_func.function_def)  # prints the function graph defination

print(type(concrete_func.graph))
print(concrete_func.graph)
isinstance(concrete_func.graph, tf.Graph) 

print(tf.autograph.to_code(add.python_function))


# use tf.function as a function(pass a func as its parameter)
v = tf.Variable(5.) 
read_and_decrement = tf.function(lambda: v.assign_sub(0.1)) 
read_and_decrement() 
print(v)