
# __init__.py 在包被导入时会被执行。
print("this package:algorithm has been imported.")


# __all__ 关联了一个模块列表，当执行 from algorithm import * 时，就会导入列表中的模块。
__all__ = ['mcts']