###################### This chatGPT API first asks ChatGPT to learn and mimic the unique writing style of the input text
###################### and then ask to generates a new text. Thus, the newly generated text should inherit the distinctive
###################### writing style of the original text.

###################### References ######################
# https://fortelabs.com/blog/how-to-create-an-ai-style-guide-write-with-chatgpt-in-your-own-voice/
# https://www.makeuseof.com/how-to-train-chatgpt-to-write-like-you/
# https://www.listendata.com/2023/05/chatgpt-in-r.html#r_function_to_allow_chatgpt_to_remember_prior_conversations
###################### References ######################

library(httr)
library(stringr)
library(jsonlite)
library(readr)
library(tools)
library(tokenizers)
source('./chatgpt_functions.R')

###################### main script ######################
###################### main script ######################
###################### main script ######################

## parameters
## parameters
apikey <- "INSET API KEY"
n.of.training.files<-3           # The number of the training texts
top.p.value<-0.1                 # Between 0 and 1 for creativity
# gpt.model<-"gpt_35-16k"          # GPT model: "gpt_35_16k" or "gpt_40"
gpt.model<-"gpt_40"
simple.or.complex<-"complex"     # either "simple" or "complex"
number.of.texts.to.generate<-50
## parameters
## parameters

info.file<-"../author_ids_three_training_topics_two_generation_topics.txt"
info.table<-data.frame(read.table(info.file,header=TRUE))

list.of.directories<-as.matrix(info.table)[1:number.of.texts.to.generate,1]

n.of.texts.already.processed<-0

for (author in list.of.directories) {

    target.all.files<-list.files(paste("../amazon_product_data_corpus_mixed_topics_per_author_reformatted/",
                                       author, sep=""),
                                 pattern='.txt', all.files=TRUE, full.names=TRUE)

    target.info.table.row<-info.table[which(info.table[,1] == author),]

    t.topic1<-as.character(target.info.table.row$training1)
    t.topic2<-as.character(target.info.table.row$training2)
    t.topic3<-as.character(target.info.table.row$training3)

    g.topic1<-as.character(target.info.table.row$generation1)

    target.all.files.before.sampling<-file_path_sans_ext(basename(target.all.files))

    # all topics included in the target directory
    all.topics.in.directory<-unlist(str_split(file_path_sans_ext(basename(target.all.files.before.sampling)),
                                                                  "_", simplify=TRUE)[,2])

    if (n.of.training.files == 1) {
        selected.topics<-c(t.topic1)
        selected.topics.collapsed<-paste(selected.topics, collapse="_")
        target.files<-paste("../amazon_product_data_corpus_mixed_topics_per_author_reformatted/",
                            author, "/", author, "_", selected.topics, ".txt", sep="")
        
    } else if (n.of.training.files == 2) {
        selected.topics<-c(t.topic1,t.topic2)
        selected.topics.collapsed<-paste(selected.topics, collapse="_")
        target.files<-paste("../amazon_product_data_corpus_mixed_topics_per_author_reformatted/",
                            author, "/", author, "_", selected.topics, ".txt", sep="")
        
    } else if (n.of.training.files == 3) {
        selected.topics<-c(t.topic1,t.topic2,t.topic3)
        selected.topics.collapsed<-paste(selected.topics, collapse="_")
        target.files<-paste("../amazon_product_data_corpus_mixed_topics_per_author_reformatted/",
                            author, "/", author, "_", selected.topics, ".txt", sep="")

    } else {}

    g.topic1<-g.topic1

    if (g.topic1 == "CellPhonesandAccessories") {
        g.topic1.prompt<-"Cell Phones and Accessories"
    } else if (g.topic1 == "ClothingShoesandJewelry") {
        g.topic1.prompt<-"Clothing, Shoes and Jewelry"
    } else if (g.topic1 == "HealthandPersonalCare") {
        g.topic1.prompt<-"Health and Personal Care"
    } else if (g.topic1 == "HomeandKitchen") {
        g.topic1.prompt<-"Home and Kitchen"
    } else if (g.topic1 == "InstantVideo") {
        g.topic1.prompt<-"Instant Video"
    } else if (g.topic1 == "MoviesandTV") {
        g.topic1.prompt<-"Movies and TV"
    } else if (g.topic1 == "GroceryandGourmetFood") {
        g.topic1.prompt<-"Grocery and Gourmet Food"
    } else if (g.topic1 == "OfficeProducts") {
        g.topic1.prompt<-"Office Products"
    } else if (g.topic1 == "DigitalMusic") {
        g.topic1.prompt<-"Digital Music"
    } else if (g.topic1 == "CDandVinyl") {
        g.topic1.prompt<-"CD and Vinyl"
    } else if (g.topic1 == "AppsforAndroid") {
        g.topic1.prompt<-"Apps for Android"
    } else if (g.topic1 == "MusicalInstruments") {
        g.topic1.prompt<-"Musical Instruments"
    } else if (g.topic1 == "KindleStore") {
        g.topic1.prompt<-"Kindle Store"
    } else {g.topic1.prompt<-g.topic1}

    topics<-paste("training_topics", selected.topics.collapsed,
                  "target_topic", g.topic1, sep="_")

    if (simple.or.complex == "simple") {
        existing.file<-paste('../chatgpt_generated_texts/simple/chatgpt_',
                               author, '_gpt_model_', gpt.model, '_top_p_', top.p.value, '_', topics, '.txt', sep="")
        
    } else if (simple.or.complex == "complex") {
        existing.file<-paste('../chatgpt_generated_texts/complex/chatgpt_',
                               author, '_gpt_model_', gpt.model, '_top_p_', top.p.value, '_', topics, '.txt', sep="")

    }

    if (file.exists(existing.file)) {
        print(paste("Skipping the author: ", author, sep=""))
        n.of.texts.already.processed<-n.of.texts.already.processed + 1
        next

    } else {
        print(paste("GPT mode: ", gpt.model, sep=""))
        print(paste("Author: ", author, sep=""))
        print(paste("Top_p: ", top.p.value, sep=""))
        print(paste("Training and target topics: ", topics, sep=""))
        print(paste("Simple or complex instruction: ", simple.or.complex, sep=""))
        n.of.texts.already.processed<-n.of.texts.already.processed + 1
        print(paste("This is the Nth file of this kind: ", n.of.texts.already.processed, sep=""))

        all.paragraphs<-c()
        for (file.to.read in target.files) {
            paragraph<-readLines(file.to.read, warn=FALSE)
            all.paragraphs<-c(all.paragraphs,paragraph)

        }

        training.texts<-paste(all.paragraphs, collapse=" ")

        ### Simple prompt ###############################
        ### Simple prompt ###############################
        main.part.simple<-paste("Study the writing style of TEXT1 provided below in order to mimic the author's writing style. ", 
                                "What did you study about the author's writing style in TEXT1?", sep="")

        training.part<- paste("TEXT1 is: ", training.texts, sep="")
        instructions1.simple<-paste(main.part.simple, training.part, sep='\n\n')

        instructions2.simple<-paste("Precisely mimick the author's unique writing style that was studied in the last prompt, ",
                             "and write a negative or positive review on any one product or service ", 
                             "belonging to the category of \'", g.topic1.prompt, 
                             "\' for more than 800 words. Do not include images. Immediately start writing the review.", sep="")

        ### Complex prompt ###############################
        ### Complex prompt ###############################
        main.part.complex<-paste("Study the writing style of TEXT1 provided below. ",
                             "Study the tone, word choice, mannerism, sentence structure, pacing, explanation style, ",
                             "the choices of punctuations & special characters, the use of upper & lower case characters ",
                             "and other stylistic elements in order to mimic this author's unique writing style. ",
                             "What did you study about the author's writing style in TEXT1?", sep="")

        instructions1.complex<-paste(main.part.complex, training.part, sep='\n\n')

        instructions2.complex<-paste("Precisely mimick the author's unique writing style that was studied in the last prompt, ",
                             "and write a negative or positive review on any one product or service ", 
                             "beloing to the category of \'", g.topic1.prompt, 
                             "\' for more than 800 words. ",
                             "Emphasize the author's distinctive writing style, including word choice, sentence strucutre, ", 
                             "format structure and the use of upper/lower case characters, punctuation marks ",
                             "and special characters. Try to mimic their tone such as use of sarcasm and humour. ",
                             "Please forgo grammatical rules and accuracies in order to abide by the author's writing style. ",
                             "Do not include images. Immediately start writing the review.", sep="")

        ### Save ChatGPT answer ###############################
        ### Save ChatGPT answer ###############################
        
        chatHistory <- list()
        if (simple.or.complex == "simple") {
            out.text<-save.chatgpt.texts.two.shot(apikey, instructions1.simple, instructions2.simple,
                                                  gpt.model, top.p.value, author, topics)
            print("################################################################################")
            print(out.text[1]); cat('\n'); print(out.text[2])
            print("################################################################################")

            write.file.name<-paste('../chatgpt_generated_texts/simple_study/chatgpt_',
                                   author, '_gpt_model_', gpt.model, '_top_p_', top.p.value, '_', topics, '.txt', sep="")
            writeLines(out.text[1], write.file.name)

            write.file.name<-paste('../chatgpt_generated_texts/simple/chatgpt_',
                                   author, '_gpt_model_', gpt.model, '_top_p_', top.p.value, '_', topics, '.txt', sep="")
            writeLines(out.text[2], write.file.name)
            
        } else if (simple.or.complex == "complex") {
            out.text<-save.chatgpt.texts.two.shot(apikey, instructions1.complex, instructions2.complex,
                                                  gpt.model, top.p.value, author, topics)
            print("################################################################################")
            print(out.text[1]); cat('\n'); print(out.text[2])
            print("################################################################################")

            write.file.name<-paste('../chatgpt_generated_texts/complex_study/chatgpt_',
                                   author, '_gpt_model_', gpt.model, '_top_p_', top.p.value, '_', topics, '.txt', sep="")
            writeLines(out.text[1], write.file.name)

            write.file.name<-paste('../chatgpt_generated_texts/complex/chatgpt_',
                                   author, '_gpt_model_', gpt.model, '_top_p_', top.p.value, '_', topics, '.txt', sep="")
            writeLines(out.text[2], write.file.name)


        }
        
    }
}
