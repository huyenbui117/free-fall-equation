from evaluate import evalulate
def main():
    config = {"degree":[], "learning_rate":[], "iterations": []}
    max_degree = 10
    learning_rate_list = [0.03]
    iterations_list = [1000, 10000]
    for d in range(max_degree):
        for lr in learning_rate_list:
            for it in iterations_list:
                config["degree"].append(d)
                config["learning_rate"].append(lr)
                config["iterations"].append(it)
    print(len(config["degree"]))
    evalulate(config=config)
if __name__ == '__main__':
    main()