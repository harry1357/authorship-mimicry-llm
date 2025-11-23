###################### This chatGPT API first asks ChatGPT to learn and mimic the unique writing style of the input text
###################### and generates a new text. Thus, the newly generated text should inherit the distinctive
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
# library(readtext)

###################### topics ######################
###################### topics ######################
###################### topics ######################

# "Beauty"                   "CellPhonesandAccessories"
# "ClothingShoesandJewelry"  "Electronics"             
# "HealthandPersonalCare"    "HomeandKitchen"          
# "InstantVideo"             "KindleStore"             
# "MoviesandTV"              "Automotive"              
# "GroceryandGourmetFood"    "OfficeProducts"          
# "Baby"                     "DigitalMusic"            
# "CDandVinyl"               "AppsforAndroid"          
# "MusicalInstruments"

all.seventeen.topics<-c("Beauty", "CellPhonesandAccessories",
                        "ClothingShoesandJewelry", "Electronics",
                        "HealthandPersonalCare", "HomeandKitchen",
                        "InstantVideo", "KindleStore",
                        "MoviesandTV", "Automotive",
                        "GroceryandGourmetFood", "OfficeProducts",
                        "Baby", "DigitalMusic",
                        "CDandVinyl", "AppsforAndroid",
                        "MusicalInstruments")


###################### function: chatGPT ######################
###################### function: chatGPT ######################
###################### function: chatGPT ######################

chatGPT <- function(apiKeyy,
                    prompt, 
                    modelName = "gpt-3.5-turbo",
                    max_tokens = 1000,
                    top_p = 0.5) {

    # Parameters
    params <- list(
        model = modelName,
        max_tokens = max_tokens,
        top_p = top_p
    )
  
    # Add the new message to the chat session messages
    chatHistory <<- append(chatHistory, list(list(role = "user", content = prompt)))
    # Provide your own API_KEY

    
    response <- POST(
      url = "https://api.openai.com/v1/chat/completions",
      add_headers("Authorization" = paste("Bearer", apiKeyy)),
      content_type_json(),
      body = toJSON(c(params, list(messages = chatHistory)), auto_unbox = TRUE)
    )
    
    if (response$status_code > 200) {
        stop(content(response))
    }
    
    response <- content(response)
    # print(params)
    answer <- trimws(response$choices[[1]]$message$content)
    chatHistory <<- append(chatHistory, list(list(role = "assistant", content = answer)))

    return(answer)
  
}

###################### function: save.chatgpt.texts.one.shot ######################
###################### function: save.chatgpt.texts.one.shot ######################
###################### function: save.chatgpt.texts.one.shot ######################
###################### This function saves the output of ChatGPT as a text file

save.chatgpt.texts.one.shot<-function(APIKEY,
                                      instructions1,
                                      # instructions2,
                                      model.name,
                                      top.p.value,
                                      authorid,
                                      topics) {

    if (gpt.model == "gpt_35") {
        gpt.model.name<-"gpt-3.5-turbo"
    } else if (gpt.model == "gpt_35-16k") {
        gpt.model.name<-"gpt-3.5-turbo-16k"
    } else if (gpt.model == "gpt_40") {
        gpt.model.name<-"gpt-4"
    } else {}

    chatGPT.answers<-c("")
    while(nchar(chatGPT.answers, type="bytes") < 3990) {
        chatGPT.answers<-c("")
        chatHistory <- list()
        chatGPT.answer1<-chatGPT(APIKEY,
                                 instructions1,
                                modelName=gpt.model.name,
                                top_p = top.p.value)

        if (nchar(chatGPT.answer1, type="bytes") < 3990) {
            chatGPT.answer2<-chatGPT(APIKEY,
                                     "Continue writing",
                                     modelName=gpt.model.name,
                                     top_p = top.p.value)

            chatGPT.answers<-paste(chatGPT.answer1, chatGPT.answer2, collapse=" ")
            print(nchar(chatGPT.answers, type="bytes"))
            print("Continue writing")

        } else {
            chatGPT.answers<-chatGPT.answer1
            print(nchar(chatGPT.answers, type="bytes"))
            print("No continue writing")
        }

        # print(chatGPT.answer1)
        # chatGPT.answer<-chatGPT(instructions2, top_p = top.p.value)

        final.output<-str_replace_all(chatGPT.answers, "\n\n" , " ")
        print("###################################################")
        print(nchar(chatGPT.answers, type="bytes"))

    }

    list.of.output<-unlist(tokenize_sentences(final.output))
    output.text<-NULL

    for (i in 1:length(list.of.output)) {
        output.text<-paste(output.text, list.of.output[i], sep=" ")
        limit<-nchar(output.text, type = "bytes")
        if (limit >= 3950) {break}

    }

    output.text<-trimws(output.text, which="left", whitespace=" ")
    write.file.name<-paste('../chatgpt_generated_texts/one/chatgpt_',
                           authorid, '_gpt_model_', gpt.model, '_top_p_', top.p.value, '_', topics, '.txt', sep="")
    writeLines(output.text, write.file.name)
    # print("################################################################################")
    return(output.text)

}

###################### function: save.chatgpt.texts.two.shot ######################
###################### function: save.chatgpt.texts.two.shot ######################
###################### function: save.chatgpt.texts.two.shot ######################
###################### This function saves the output of ChatGPT as a text file

save.chatgpt.texts.two.shot<-function(APIKEY,
                                      instructions1,
                                      instructions2,
                                      model.name,
                                      top.p.value,
                                      authorid,
                                      topics) {

    if (gpt.model == "gpt_35") {
        gpt.model.name<-"gpt-3.5-turbo"
    } else if (gpt.model == "gpt_35-16k") {
        gpt.model.name<-"gpt-3.5-turbo-16k"
    } else if (gpt.model == "gpt_40") {
        gpt.model.name<-"gpt-4"
    } else if (gpt.model == "gpt_40-1106") {
        gpt.model.name<-"gpt-4-1106-preview"
    } else {}

    chatGPT.answers<-c("")

    while(nchar(chatGPT.answers, type="bytes") < 3990) {
        chatGPT.answers<-c("")
        chatHistory <- list()
        
        chatGPT.answer.intro<-chatGPT(APIKEY,
                                      instructions1,
                                      modelName=gpt.model.name,
                                      top_p = top.p.value)
        
        if (str_detect(unlist(tokenize_sentences(chatGPT.answer.intro))[1], "TEXT1|writing style")) {
            # do nothing
        } else {
            print(chatGPT.answer.intro)
            chatGPT.answers<-"The output is not long enough or bad."
            break
        }

        chatGPT.answer1<-chatGPT(APIKEY,
                                 instructions2,
                                 modelName=gpt.model.name,
                                 top_p = top.p.value)

        if (nchar(chatGPT.answer1, type="bytes") < 3990) {
            chatGPT.answer2<-chatGPT(APIKEY,
                                     "Following the same insturctions given in the last prompt, continue writing for more than 400 words",
                                     modelName=gpt.model.name,
                                     top_p = top.p.value)

            chatGPT.answers<-paste(chatGPT.answer1, chatGPT.answer2, collapse=" ")
            print(nchar(chatGPT.answers, type="bytes"))
            print("Continue writing")

        } else {
            chatGPT.answers<-chatGPT.answer1
            print(nchar(chatGPT.answers, type="bytes"))
            print("Not continue writing")
        }

        if (nchar(chatGPT.answers, type="bytes") < 3990) {
            chatGPT.answers<-"The output is not long enough or bad."
            final.output<-str_replace_all(chatGPT.answers, "\n\n" , " ")
            # print("###################################################")
            # print(nchar(chatGPT.answers, type="bytes"))

            break
            
        } else {
            final.output<-str_replace_all(chatGPT.answers, "\n\n" , " ")
            # print("###################################################")
            # print(nchar(chatGPT.answers, type="bytes"))

        }

    }

    if (chatGPT.answers == "The output is not long enough or bad.") {
        return(c(chatGPT.answers, chatGPT.answers))

    } else {

        list.of.output<-unlist(tokenize_sentences(final.output))
        output.text<-NULL
    
        for (i in 1:length(list.of.output)) {
            output.text<-paste(output.text, list.of.output[i], sep=" ")
            limit<-nchar(output.text, type = "bytes")
            if (limit >= 3950) {break}

        }

        output.text<-trimws(output.text, which="left", whitespace=" ")
        # write.file.name<-paste('../chatgpt_generated_texts/simple/chatgpt_',
        #                   authorid, '_gpt_model_', gpt.model, '_top_p_', top.p.value, '_', topics, '.txt', sep="")
        # writeLines(output.text, write.file.name)
        # print("################################################################################")
        return(c(chatGPT.answer.intro, output.text))

    }
}

