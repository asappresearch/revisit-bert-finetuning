import pandas as pd
import os
import time

start_time = time.time()

results_walk = os.walk("bert_output")

results_df = pd.DataFrame([])

for file_tuple in results_walk:
    print(file_tuple)
    path = file_tuple[0]
    if "SEED" in path:
        seed = int(path[path.rindex("D") + 1 :])
        subpath = path[: path.rindex("/")]
        dataset = subpath[subpath.rindex("/") + 1 :]
        subpath = subpath[: subpath.rindex("/")]
        method = subpath[subpath.rindex("/") + 1 :]

        for filename in file_tuple[2]:
            if filename[-4:] == ".txt":
                filepath = file_tuple[0] + "/" + filename
                df = pd.read_csv(filepath)

                df["i"] = [i for i in range(len(df))]
                df["filetype"] = [filename[:-4]] * len(df)
                df["seed"] = [seed] * len(df)
                df["dataset"] = [dataset] * len(df)
                df["method"] = [method] * len(df)
                results_df = pd.concat([results_df, df], axis=0, ignore_index=True)

results_df.to_csv("replicate_results.csv")

print("--- %s seconds ---" % (time.time() - start_time))
