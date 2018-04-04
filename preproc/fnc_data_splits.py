import random
from csv import DictReader
from csv import DictWriter

# Define data class
class FNCData:

    """
    Define class for Fake News Challenge data
    """

    def __init__(self, file_instances, file_bodies):

        # Load data
        self.instances = self.read(file_instances)
        bodies = self.read(file_bodies)
        self.heads = {}
        self.bodies = {}

        # Process instances
        for instance in self.instances:
            if instance['Headline'] not in self.heads:
                head_id = len(self.heads)
                self.heads[instance['Headline']] = head_id
            instance['Body ID'] = int(instance['Body ID'])

        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['articleBody']

    def read(self, filename):

        """
        Read Fake News Challenge data from CSV file
        Args:
            filename: str, filename + extension
        Returns:
            rows: list, of dict per instance
        """

        # Initialise
        rows = []

        # Process file
        with open(filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows


def split_seen(data, rand=False, prop_dev=0.2, rnd_sd=1489215):

    """

    Split data into separate sets with overlapping headlines

    Args:
        data: FNCData object
        rand: bool, True: random split and False: use seed for official baseline split
        prop_dev: float, proportion of data for dev set
        rnd_sd: int, random seed to use for split

    Returns:
        train: list, of dict per instance
        dev: list, of dict per instance

    """

    # Initialise
    list_bodies = [body for body in data.bodies]
    n_dev_bodies = round(len(list_bodies) * prop_dev)
    r = random.Random()
    if rand is False:
        r.seed(rnd_sd)
    train = []
    dev = []

    # Generate list of bodies for dev set
    r.shuffle(list_bodies)
    list_dev_bodies = list_bodies[-n_dev_bodies:]

    # Generate train and dev sets
    for stance in data.instances:
        if stance['Body ID'] not in list_dev_bodies:
            train.append(stance)
        else:
            dev.append(stance)

    return train, dev


def split_unseen(data, rand=False, prop_dev=0.2, rnd_sd=1489215):

    """

    Split data into completely separate sets (i.e. non-overlap of headlines and bodies)

    Args:
        data: FNCData object
        rand: bool, True: random split and False: constant split
        prop_dev: float, target proportion of data for dev set
        rnd_sd: int, random seed to use for split

    Returns:
        train: list, of dict per instance
        dev: list, of dict per instance

    """

    # Initialise
    n = len(data.instances)
    n_dev = round(n * prop_dev)
    dev_ind = {}
    r = random.Random()
    if rand is False:
        r.seed(rnd_sd)
    train = []
    dev = []

    # Identify instances for dev set
    while len(dev_ind) < n_dev:
        rand_ind = r.randrange(n)
        if not data.instances[rand_ind]['Stance'] in ['agree', 'disagree', 'discuss']:
            continue
        if rand_ind not in dev_ind:
            rand_head = data.instances[rand_ind]['Headline']
            rand_body_id = data.instances[rand_ind]['Body ID']
            dev_ind[rand_ind] = 1
            track_heads = {}
            track_bodies = {}
            track_heads[rand_head] = 1
            track_bodies[rand_body_id] = 1
            pre_len_heads = len(track_heads)
            pre_len_bodies = len(track_bodies)
            post_len_heads = 0
            post_len_bodies = 0
            while pre_len_heads != post_len_heads and pre_len_bodies != post_len_bodies:
                pre_len_heads = len(track_heads)
                pre_len_bodies = len(track_bodies)
                for i, stance in enumerate(data.instances):
                    if not data.instances[i]['Stance'] in ['agree', 'disagree', 'discuss']:
                        continue
                    if i != rand_ind and (stance['Headline'] in track_heads or stance['Body ID'] in track_bodies):
                        track_heads[stance['Headline']] = 1
                        track_bodies[stance['Body ID']] = 1
                post_len_heads = len(track_heads)
                post_len_bodies = len(track_bodies)

            for k, stance in enumerate(data.instances):
                if k != rand_ind and (stance['Headline'] in track_heads or stance['Body ID'] in track_bodies) and (stance['Stance'] in ['agree', 'disagree', 'discuss']):
                    dev_ind[k] = 1

    # Generate train and dev sets
    for k, stance in enumerate(data.instances):
        if k in dev_ind:
            dev.append(stance)
        else:
            train.append(stance)

    return train, dev


def save_csv(data_split, filepath):
    """
    Save predictions to CSV file
    Args:
        pred: numpy array, of numeric predictions
        file: str, filename + extension
    """

    with open(filepath, 'w', encoding='utf-8') as csvfile:
        fieldnames = ['Headline','Body ID','Stance']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for instance in data_split:
            writer.writerow({'Headline': instance["Headline"], 'Body ID': instance["Body ID"], 'Stance': instance["Stance"]})


if __name__ == "__main__":
    data = FNCData("../data/fakenewschallenge/train_stances.csv", "../data/fakenewschallenge/train_bodies.csv")
    train, dev = split_unseen(data)
    save_csv(train, "../data/fakenewschallenge/trainsplit_stances.csv")
    save_csv(dev, "../data/fakenewschallenge/devsplit_stances.csv")