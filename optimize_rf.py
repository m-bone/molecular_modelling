
import numpy as np
import pandas as pd
import pygad
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import create_connection

seed = 3

# Load data from PostgreSQL database
engine = create_connection()
desc_df = pd.read_sql_table('gdb9_descriptors_full', engine)
prop_df = pd.read_sql_table('gdb9_properties_full', engine)

# Clip DFs for testing
desc_df = desc_df.iloc[:5000, :]
prop_df = prop_df.iloc[:5000, :]

prop_target = "dipole_moment"
target_value = 3.0

# Remove columns with no variance
desc_df = desc_df.loc[:, desc_df.var() != 0]

# Remove gdb_tag column
desc_df = desc_df.drop(columns=['gdb_tag'])

targets = np.where(prop_df[prop_target] > target_value, 1, 0)

# Previous solutions
prev_sol = {}

# Define the optimizer
# Define all required variables and functions
def fitness_function(ga_instance: pygad.GA, solution: np.array, solution_idx):
    """"""
    acc_list = []
    for sol in solution:

        # Convert solution to idx of selected features
        selected_features = np.where(sol > 0.5)[0]

        # Check if the solution has been evaluated before
        prev_sol_key = ",".join(map(str, selected_features))

        if prev_sol_key in prev_sol:
            acc_test = prev_sol[prev_sol_key]
            print(f"Using previous solution")
            acc_list.append(acc_test)
            
        else:
            sol_desc_df = desc_df.iloc[:, selected_features]
            print(f"Number of features selected: {len(selected_features)}")

            # Perform fit and evaluate the model
            X_train, X_test, y_train, y_test = train_test_split(sol_desc_df, targets, test_size=0.2, random_state=seed)

            clf = BalancedRandomForestClassifier(n_estimators=100, random_state=seed)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc_test = accuracy_score(y_test, y_pred)
            acc_list.append(acc_test)

            # Update previous solutions to prevent duplicate calculations
            prev_sol[prev_sol_key] = acc_test

    return acc_list

def on_generation(ga_instance):
    # Announce best fitness at each generation
    best_fitness = ga_instance.best_solution()[1]
    print(
        f"Generation {ga_instance.generations_completed} Best Fitness: {best_fitness:.3f}"
    )

# Genetic Algorithm parameters
num_generations = 50
num_parents_mating = 12           
parent_selection_type = "tournament"  
keep_parents = 2
mutation_type = "random"
mutation_probability = 0.05
crossover_type = "uniform"
crossover_probability = 0.8
solutions_per_pop = 25

num_genes = desc_df.shape[1]
fitness_batch_size = solutions_per_pop
random_seed = seed
gene_space = [{'low': 0, 'high': 1}] * num_genes
gene_type = float
stop_criteria = ["reach_100"]
save_solutions = True

# Create the Genetic Algorithm instance
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    mutation_type=mutation_type,
    mutation_probability=mutation_probability,
    crossover_type=crossover_type,
    crossover_probability=crossover_probability,
    sol_per_pop=solutions_per_pop,
    num_genes=num_genes,
    fitness_batch_size=fitness_batch_size,
    on_generation=on_generation,
    random_seed=random_seed,
    gene_space=gene_space,
    gene_type=gene_type,
    stop_criteria=stop_criteria,
    save_solutions=save_solutions,
    suppress_warnings=False,  # Used to prevent delay_on_generation depreciation warning for PyGAD 3.3.0
)

ga_instance.run()

solution, solution_fitness, _ = ga_instance.best_solution()
print(
    f"Best Result: {solution_fitness}\n"
    f"Best Parameters: TBD\n"
)

ga_instance.plot_fitness()
ga_instance.plot_new_solution_rate()
ga_instance.save("best_solution")