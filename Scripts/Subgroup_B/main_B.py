import pickle

def mainB():
    # load the comparison results saved earlier
    with open("b_qn2_comparison.pkl", "rb") as f:
        comp = pickle.load(f)

    # print current layout
    print("\nQuestion 3: Optimization of Attraction Layout and Schedules")
    print("\nCurrent USS Layout (Two Entrances) - Multi Queue:")
    for attraction, time in comp["avg_wait_times_1_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_1']:.2f} min")
    print("Visit Counts:", comp["visit_counts_1_multi"])

    # print modified layout to show time difference
    print("\nModified USS Layout (Left Entrance Only, Swapped Transformers and CYLON) - Multi Queue:")
    for attraction, time in comp["avg_wait_times_2_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_2']:.2f} min")
    print("Visit Counts:", comp["visit_counts_2_multi"])

if __name__ == "__main__":
    mainB()