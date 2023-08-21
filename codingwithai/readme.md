Lets Code with AI assistants like chatgpt and llama and see what happens

* Various python and c programs
* Solving problems that may be daunting to code 
* Building better AI-Human workflows



## CTP
Basic CAN transport protocol with tests.

I made this using GPT-4 code interpreter.

This was a pretty interesting example, I allowed gpt to do almost all the work with me reviewing
and asking the right questions. I did make some modifications along the way on a few things,
but 95% was done with only prompting.

**I structured the prompts as follows.**

1. Can you devise a solution to X
2. Can you write this solution in C code
3. Can you write tests for this code
4. Okay the tests aren't quite working can you revise the code  

  
    
**I started off the prompt chain as follows**

1. Can you devise a transport protocol for a CAN network?
2. Okay, can you write this in C code?
3. Can you write tests for this C code?
4. Great, there are some missing functionality can you fill in function X (repeat for anything missing)
5. Some tests aren't passing can you figure out why?

### Notes
The exiting thing was at most steps the code was actually working code and almost complete code.  
I had to make some improvements to the code for example:
  
1. Since I broke these problems up into parts I had to the the copy paste, build and organization of the code.
2. There was one bug generated that I found with a debugger, an off by one bug.

I ended up spending probably an hour or two prompting back and forth and reviewing to write this code.
The cool part was the workflow, it seemed very smooth and intuitive. I spent the most time reviewing and working out the mock and test code with the system.

A couple things I noticed while AI programming:
  
1. GPT was able to start off simple and build into complexity. At times as an engineer working on a given problem, I might overthink things or try to make a solution too complex too quickly, instead of starting with a basic example and building up. GPT broke the problem down into logical parts and started with a basic example, I was then able to have it build up the missing parts and finish out everything that was missing. 
2. It understood very well what I wanted. I clearly asked it about a physical layer and a transport layer and it was able to understand I wanted a workable transport layer for the given physical layer.
3. It wrote working code and it wrote tests for the code even though there is some complexities with mocking hardware.
4. There was an off by one bug in the code, it took me some review to find, I commented the line at !!Note
5. There was some amount of time spent looking at and understanding how the code worked, because the GPT at written things that I didn't quite understand and I thought were bugs but after some review and validation from tests I realized it was correct.
6. I was able to mislead GPT into making redundant code, there were parts where I thought there may be an issue and asked the agent about it and it would generate code, but after review I realized the original code was correct.
7. There was parts that I was unable to trick GPT, I was able to to ask it a question on a given line and it responded it was sure the line was correct.
8. Some prompts it didn't solve directly but layed out a guide on how to solve the problem and gave debug code.

### Conclusions and Improvements

The AI-human pair programmer model is really good, I'll continue to do this and integrate agents into my programming
as much as possible. Obviously this is a toy example but I'm convinced that the productivity and the value of having help 
with coding and debugging is really valuable.

The file upload feature is really powerful, being able to upload a file and work from there is super powerful and the ability 
to breakdown data is awesome. I need to figure out how to get more common formats working well like PDF for example.

The code it produced isn't what you would see in a production system, however I think with extra time and some improvements
it could certainly get there.