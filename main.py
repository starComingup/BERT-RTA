from torch_dataset.AmazonDataset import AmazonDataset
from torch_dataset.FakeJobDataset import FakeJobDataset
from torch_dataset.FinancialDataset import FinancialDataset
from torch_dataset.NewsDataset import NewsDataset
from torch_dataset.RedditDataset import RedditDataset
from torch_dataset.SpamDataset import SpamDataset
from torch_dataset.TwitterDataset import TwitterDataset
from torch_dataset.YoutubeDataset import YoutubeDataset
from torch_dataset.TwitterNDataset import TwitterNDataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tokenizer = None
    max_len = 128

    # amazonDataset = AmazonDataset(tokenizer, max_len)
    # print(amazonDataset.get_class_counts())
    # for text in amazonDataset.texts[:25]:
    #     print(text)
    # for label in amazonDataset.labels[:25]:
    #     print(label)
    # counts = amazonDataset.labels.value_counts()
    # 输出类别和对应的数量
    # for category, count in counts.items():
    #     print(f"类别：{category}，数量：{count}")
    # fakeJobDataset = FakeJobDataset(tokenizer, max_len)
    # for text in fakeJobDataset.texts[:5]:
    #     print(text)
    # for label in fakeJobDataset.labels[:5]:
    #     print(label)

    # financialDataset = FinancialDataset(tokenizer, max_len)
    # print(financialDataset.get_class_counts())
    # for text in financialDataset.texts[:5]:
    #     print(text)
    # for label in financialDataset.labels[:5]:
    #     print(label)

    # newsDataset = NewsDataset(tokenizer, max_len)
    # for text in newsDataset.texts[:5]:
    #     print(text)
    # for label in newsDataset.labels[:5]:
    #     print(label)

    # redditDataset = RedditDataset(tokenizer, max_len)
    # print(redditDataset.get_class_counts())
    # for text in redditDataset.texts[:5]:
    #     print(text)
    # for label in redditDataset.labels[:5]:
    #     print(label)

    # spamDataset = SpamDataset(tokenizer, max_len)
    # for text in spamDataset.texts[:5]:
    #     print(text)
    # for label in spamDataset.labels[:5]:
    #     print(label)

    # spamDataset = SpamDataset(tokenizer, max_len)
    # for text in spamDataset.texts[:5]:
    #     print(text)
    # for label in spamDataset.labels[:5]:
    #     print(label)

    # twitterDataset = TwitterDataset(tokenizer, max_len)
    # print(twitterDataset.get_class_counts())
    # print(len(twitterDataset))
    # for text in twitterDataset.texts[:5]:
    #     print(text)
    # for label in twitterDataset.labels[:5]:
    #     print(label)

    youtubeDataset = YoutubeDataset(tokenizer, max_len)
    print(youtubeDataset.get_class_counts())




