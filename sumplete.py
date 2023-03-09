import functools
import itertools
import operator
import copy
import sys
from multiprocessing.pool import Pool
from tqdm import tqdm

def pretty_print(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
    print()


grid = [[2, 9, 3],
        [5, 3, 4],
        [7, 7, 7],
        ]
rowTargets = [11, 9, 14]
colTargets = [14, 16, 4]

grid = [[7, 2, 9, 8],
        [2, 7, 5, 5],
        [9, 7, 2, 4],
        [9, 8, 6, 4]]
rowTargets = [7, 17, 4, 27]
colTargets = [16, 15, 11, 13]

grid = [[2, 8, 7, 4, 8, 7, 5],
        [1, 1, 8, 5, 3, 7, 1],
        [1, 4, 5, 9, 7, 5, 1],
        [6, 1, 7, 1, 5, 5, 3],
        [4, 6, 1, 9, 4, 5, 3],
        [6, 2, 5, 8, 4, 9, 8],
        [7, 3, 2, 5, 9, 7, 3]]
rowTargets = [26, 14, 11, 15, 16, 14, 17]
colTargets = [12, 17, 30, 18, 15, 19, 2]
pretty_print(grid)

grid = [[9, 8, 3, 2, 8, 3, 4, 2],
        [4, 2, 6, 9, 2, 4, 1, 6],
        [6, 9, 7, 7, 2, 5, 8, 4],
        [2, 8, 3, 1, 9, 3, 6, 4],
        [1, 4, 2, 9, 5, 1, 2, 3],
        [2, 2, 6, 6, 8, 4, 8, 3],
        [4, 8, 6, 4, 7, 1, 6, 9],
        [9, 4, 9, 7, 6, 3, 2, 5]]
rowTargets = [33, 27, 30, 8, 21, 35, 29, 26]
colTargets = [27, 37, 24, 30, 31, 4, 26, 30]

grid = [[-15, -11, 1, 19, 18, 14, 12, -13, -9],
        [12, 17, -19, -14, -13, -7, -18, -8, 7],
        [19, -3, -7, 7, 19, 5, -7, -16, 2],
        [7, 9, 12, -10, -6, 8, -19, 15, 17],
        [12, 7, 2, -10, 17, 8, 12, 14, -8],
        [-12, 13, 10, 9, -5, -10, -14, -18, -4],
        [-17, -4, -3, 2, 7, 10, -2, -16, -1],
        [5, -19, 17, 1, -12, 13, -2, -4, 2],
        [11, 1, -6, -11, -12, 14, -11, -16, -14],]
rowTargets = [6, -5, 31, 42, 45, -19, -19, -1, 15]
colTargets = [4, -5, 27, 12, 2, 23, 22, 5, 5]

# Hardest so far, naive solution is 25 days, recursive is 7 minutes
grid = [[18, -14, 8, -3, 13, -17, -7, -17, -5],
        [-19, 16, -16, -4, -19, -12, -11, 12, 17],
        [-19, -14, 3, -14, 1, 16, -19, 1, 12],
        [-13, 3, -16, 8, 7, -8, 7, 6, 14],
        [16, 9, -12, -7, -9, 18, -4, 9, -11],
        [10, 18, -13, -11, 4, -18, -2, 10, 10],
        [-17, -18, -1, 16, 5, -8, 17, 2, -14],
        [-14, 1, -18, 14, -8, -10, 11, -17, -7],
        [-5, -6, -16, -13, 13, -9, -17, 6, -16],]
rowTargets = [-25, -6, -2, 3, 16, 7, -8, -31, -16]
colTargets = [-35, 4, -32, 2, 17, -25, -9, 10, 6]

def getSolutionSet(ls, target_sum):
    possible_to_delete = []
    for x in range(len(ls) + 1):
        # Solutions for deleting x numbers from row:
        total_sum = sum(ls)
        for y in itertools.combinations(enumerate(ls), x):
            if total_sum - sum([j for i, j in y]) == target_sum:
                possible_to_delete.append([i for i, j in y])
    return possible_to_delete


def getRowSums(grid):
    return [sum([x for x in row if x != 'X']) for row in grid]

def getColSums(grid):
    return [sum([grid[col][row] for col in range(len(grid[0])) if grid[col][row] != 'X']) for row in range(len(grid))]

def check_valid(idx, solution, set_dict):
    for item in solution:
        if all([idx not in d_set for d_set in set_dict[item]]):
            return False
    return True

def valid_grid(test_row, test_col):
    test_grid = copy.deepcopy(grid)
    values_to_delete = set()
    for row_idx, solution in test_row.items():
        [values_to_delete.add((row_idx, x)) for s in solution for x in s]
    for col_idx, solution in test_col.items():
        [values_to_delete.add((x, col_idx)) for s in solution for x in s]
    for x, y in values_to_delete:
        test_grid[x][y] = 'X'
    return getRowSums(test_grid) == rowTargets and getColSums(test_grid) == colTargets

def trim_sets(rows, cols, runs=1):
    count = 0
    for count in range(runs):
        for idx, row_sets in rows.items():
            valid_sets = [row_set for row_set in row_sets if check_valid(idx, row_set, cols)]
            rows[idx] = valid_sets
        # Check each row: delete invalid
        for idx, col_sets in cols.items():
            valid_sets = [col_set for col_set in col_sets if check_valid(idx, col_set, rows)]
            cols[idx] = valid_sets
        count += 1


class Solver:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def recurse(self, solution, bar):
        bar.update(1)
        if max([len(x) for x in solution.values()]) <= 1:
            trimmed_cols = self.cols.copy()
            trim_sets(solution, trimmed_cols, runs=1)
            if valid_grid(solution, trimmed_cols):
                return True, solution, trimmed_cols
            else:
                return False, [], []
        else:
            for i, x in sorted(solution.items(), key=lambda item: len(item[1])):  # solution.items():
                if len(x) > 1:
                    for sub_solution in x:
                        possible_solution = solution.copy()
                        possible_solution[i] = [sub_solution]
                        trimmed_cols = self.cols.copy()
                        trim_sets(possible_solution, trimmed_cols, runs=len(self.rows))
                        if valid_grid(possible_solution, trimmed_cols):
                            return True, possible_solution, trimmed_cols
                        else:
                            done, rows, cols = self.recurse(possible_solution, bar)
                            if done:
                                return True, rows, cols
        return False, [], []

    def solve(self):
        trim_sets(self.rows, self.cols, len(self.rows))
        if valid_grid(self.rows, self.cols):
            return self.rows, self.cols
        else:
            bar = tqdm()
            done, rows, cols = self.recurse(self.rows, bar=bar)
            if done:
                return rows, cols
            return None, None

def getColumns(ls, target):
    r = []
    for x in range(len(ls) + 1):
        # Solutions for deleting x numbers from row:
        total_sum = sum(ls)
        for y in itertools.combinations(enumerate(ls), x):
            k = [i for i, j in y]
            s = total_sum - sum([j for i, j in y])
            r.append((k, s))
    return r


if __name__ == '__main__':
    pretty_print(grid)

    # Initial experiments with dlx solver
    # print({f'r{row_idx}_d{"_".join(map(str, k))}': sum
    #        for (row_idx, row), target in zip(enumerate(grid), rowTargets)
    #        for k, sum in getColumns(row, target)})
    # columns = [[grid[col][row] for col in range(len(grid[0]))] for row in range(len(grid))]
    # print({f'c{col_idx}_d{"_".join(map(str, k))}': sum
    #        for (col_idx, col), target in zip(enumerate(columns), colTargets)
    #        for k, sum in getColumns(col, target)})

    possible_delete_rows = {row_idx: getSolutionSet(row, target)
                            for (row_idx, row), target in zip(enumerate(grid), rowTargets)}
    columns = [[grid[col][row] for col in range(len(grid[0]))] for row in range(len(grid))]
    possible_delete_cols = {col_idx: getSolutionSet(col, target)
                            for (col_idx, col), target in zip(enumerate(columns), colTargets)}
    solver = Solver(possible_delete_rows, possible_delete_cols)
    rows, cols = solver.solve()

    values_to_delete = set()
    for row_idx, solution in rows.items():
        [values_to_delete.add((row_idx, x)) for s in solution for x in s]
    for col_idx, solution in cols.items():
        [values_to_delete.add((x, col_idx)) for s in solution for x in s]
    for x, y in values_to_delete:
        grid[x][y] = 'X'
    pretty_print(grid)
