"""
Advanced Dataset Generator for dLNk GPT
Generates 50,000+ high-quality diverse training examples
"""

import json
import random
from typing import List, Dict
import os

class AdvancedDatasetGenerator:
    def __init__(self, target_size: int = 50000):
        self.target_size = target_size
        self.dataset = []
        
        # Diverse topic categories
        self.topics = [
            "technology", "science", "philosophy", "history", "literature",
            "mathematics", "psychology", "economics", "politics", "art",
            "medicine", "law", "education", "business", "engineering",
            "astronomy", "biology", "chemistry", "physics", "sociology",
            "anthropology", "linguistics", "music", "film", "sports",
            "cooking", "travel", "environment", "ethics", "religion"
        ]
        
        # Conversation styles
        self.styles = [
            "formal", "casual", "technical", "creative", "analytical",
            "persuasive", "descriptive", "narrative", "expository", "argumentative"
        ]
        
        # Instruction types
        self.instruction_types = [
            "explain", "analyze", "compare", "create", "summarize",
            "evaluate", "describe", "discuss", "argue", "critique"
        ]
        
    def generate_instruction_following(self, count: int) -> List[Dict]:
        """Generate instruction-following examples"""
        examples = []
        
        instruction_templates = [
            {
                "instruction": "Explain the concept of {topic} in {style} terms.",
                "response": "Let me explain {topic} in {style} terms. {topic_content} This concept is fundamental because {reasoning}. In practical applications, {application}."
            },
            {
                "instruction": "Write a {style} analysis of {topic}.",
                "response": "Analysis of {topic}: {analysis_intro} Key points include: 1) {point1} 2) {point2} 3) {point3}. In conclusion, {conclusion}."
            },
            {
                "instruction": "Compare and contrast {topic1} with {topic2}.",
                "response": "Comparing {topic1} and {topic2}: Similarities include {similarity}. However, they differ in {difference1} and {difference2}. Overall, {comparison_conclusion}."
            },
            {
                "instruction": "Create a detailed guide on {topic}.",
                "response": "Comprehensive guide to {topic}: Step 1: {step1}. Step 2: {step2}. Step 3: {step3}. Important considerations: {considerations}. Best practices: {best_practices}."
            },
            {
                "instruction": "Discuss the implications of {topic} in modern society.",
                "response": "The implications of {topic} in modern society are profound. Socially, {social_impact}. Economically, {economic_impact}. Ethically, {ethical_impact}. Looking forward, {future_outlook}."
            },
            {
                "instruction": "Provide a {style} perspective on {topic}.",
                "response": "From a {style} perspective, {topic} can be understood as {perspective_intro}. This viewpoint emphasizes {emphasis}. Critics might argue {counterpoint}, but {rebuttal}."
            },
            {
                "instruction": "Analyze the relationship between {topic1} and {topic2}.",
                "response": "The relationship between {topic1} and {topic2} is complex. {topic1} influences {topic2} through {mechanism1}. Conversely, {topic2} affects {topic1} via {mechanism2}. This bidirectional relationship {conclusion}."
            },
            {
                "instruction": "Evaluate the effectiveness of {topic} in {context}.",
                "response": "Evaluating {topic} in {context}: Strengths include {strength1} and {strength2}. Weaknesses are {weakness1} and {weakness2}. Overall effectiveness rating: {rating} because {justification}."
            },
            {
                "instruction": "Describe the historical development of {topic}.",
                "response": "The historical development of {topic} spans several eras. Early period: {early_history}. Middle period: {middle_history}. Modern era: {modern_history}. Current state: {current_state}."
            },
            {
                "instruction": "Argue for or against {topic} using {style} reasoning.",
                "response": "Argument regarding {topic}: Thesis: {thesis}. Supporting evidence: {evidence1}, {evidence2}, {evidence3}. Counterarguments: {counter}. Rebuttal: {rebuttal}. Conclusion: {conclusion}."
            }
        ]
        
        for _ in range(count):
            template = random.choice(instruction_templates)
            topic = random.choice(self.topics)
            topic2 = random.choice([t for t in self.topics if t != topic])
            style = random.choice(self.styles)
            
            # Generate varied content
            instruction = template["instruction"].format(
                topic=topic, topic1=topic, topic2=topic2, 
                style=style, context=f"{style} {random.choice(self.topics)}"
            )
            
            response = template["response"].format(
                topic=topic, topic1=topic, topic2=topic2, style=style,
                topic_content=f"a fundamental principle in {topic}",
                reasoning=f"it forms the basis of {style} understanding",
                application=f"we see this in {random.choice(self.topics)}",
                analysis_intro=f"examining {topic} through {style} lens reveals",
                point1=f"the {style} nature of {topic}",
                point2=f"its relationship to {topic2}",
                point3=f"practical implications in {random.choice(self.topics)}",
                conclusion=f"{topic} remains crucial for {style} development",
                similarity=f"both involve {style} approaches",
                difference1=f"{topic} emphasizes {random.choice(self.styles)}",
                difference2=f"{topic2} focuses on {random.choice(self.styles)}",
                comparison_conclusion=f"each serves distinct purposes in {random.choice(self.topics)}",
                step1=f"understand the {style} foundation",
                step2=f"apply {topic} principles",
                step3=f"integrate with {topic2}",
                considerations=f"ensure {style} compatibility",
                best_practices=f"maintain {style} standards",
                social_impact=f"it transforms {style} interactions",
                economic_impact=f"it influences {random.choice(self.topics)} markets",
                ethical_impact=f"it raises questions about {random.choice(self.topics)}",
                future_outlook=f"we expect {style} evolution",
                perspective_intro=f"a {style} framework for understanding",
                emphasis=f"the {style} aspects of {topic}",
                counterpoint=f"the {style} limitations",
                rebuttal=f"evidence shows {style} validity",
                mechanism1=f"{style} processes",
                mechanism2=f"{random.choice(self.styles)} dynamics",
                strength1=f"{style} applicability",
                strength2=f"robust {random.choice(self.styles)} framework",
                weakness1=f"limited {style} scope",
                weakness2=f"requires {random.choice(self.styles)} expertise",
                rating=f"{random.choice(['high', 'moderate', 'significant'])}",
                justification=f"it demonstrates {style} effectiveness",
                early_history=f"origins in {style} {random.choice(self.topics)}",
                middle_history=f"expansion through {random.choice(self.styles)} methods",
                modern_history=f"integration with {topic2}",
                current_state=f"leading {style} paradigm",
                thesis=f"{topic} represents {style} advancement",
                evidence1=f"{style} research supports this",
                evidence2=f"{topic2} demonstrates correlation",
                evidence3=f"empirical {random.choice(self.styles)} data confirms",
                counter=f"some argue {style} limitations exist",
                context=f"{style} {random.choice(self.topics)}"
            )
            
            examples.append({
                "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            })
        
        return examples
    
    def generate_qa_pairs(self, count: int) -> List[Dict]:
        """Generate question-answer pairs"""
        examples = []
        
        qa_templates = [
            {
                "question": "What is {topic} and why is it important?",
                "answer": "{topic} is {definition}. It's important because {importance}. Applications include {applications}."
            },
            {
                "question": "How does {topic} relate to {topic2}?",
                "answer": "{topic} relates to {topic2} through {relationship}. This connection is significant because {significance}."
            },
            {
                "question": "Can you explain {topic} using {style} language?",
                "answer": "In {style} terms, {topic} means {explanation}. Think of it as {analogy}."
            },
            {
                "question": "What are the main challenges in {topic}?",
                "answer": "The main challenges in {topic} include: 1) {challenge1}, 2) {challenge2}, 3) {challenge3}. Solutions involve {solutions}."
            },
            {
                "question": "How has {topic} evolved over time?",
                "answer": "{topic} has evolved significantly. Initially {past}, then {transition}, now {present}. Future trends suggest {future}."
            },
            {
                "question": "What are the ethical considerations of {topic}?",
                "answer": "Ethical considerations of {topic} include {ethical1}, {ethical2}, and {ethical3}. These require {approach}."
            },
            {
                "question": "How can {topic} be applied in {context}?",
                "answer": "Applying {topic} in {context} involves {method}. Benefits include {benefits}. Limitations are {limitations}."
            },
            {
                "question": "What distinguishes {topic} from {topic2}?",
                "answer": "{topic} differs from {topic2} in {distinction1} and {distinction2}. However, both share {commonality}."
            },
            {
                "question": "What are the prerequisites for understanding {topic}?",
                "answer": "To understand {topic}, you need {prereq1}, {prereq2}, and {prereq3}. Start with {starting_point}."
            },
            {
                "question": "What are common misconceptions about {topic}?",
                "answer": "Common misconceptions about {topic}: 1) {misconception1} - actually {truth1}. 2) {misconception2} - in reality {truth2}."
            }
        ]
        
        for _ in range(count):
            template = random.choice(qa_templates)
            topic = random.choice(self.topics)
            topic2 = random.choice([t for t in self.topics if t != topic])
            style = random.choice(self.styles)
            context = f"{random.choice(self.styles)} {random.choice(self.topics)}"
            
            question = template["question"].format(
                topic=topic, topic2=topic2, style=style, context=context
            )
            
            answer = template["answer"].format(
                topic=topic, topic2=topic2, style=style, context=context,
                definition=f"a {style} concept in {random.choice(self.topics)}",
                importance=f"it enables {style} understanding of {topic2}",
                applications=f"{random.choice(self.topics)} and {random.choice(self.topics)}",
                relationship=f"{style} connections and {random.choice(self.styles)} interactions",
                significance=f"it impacts {random.choice(self.topics)}",
                explanation=f"a {style} approach to {random.choice(self.topics)}",
                analogy=f"similar to {topic2} but with {style} characteristics",
                challenge1=f"{style} complexity",
                challenge2=f"integration with {topic2}",
                challenge3=f"maintaining {random.choice(self.styles)} standards",
                solutions=f"{style} methodologies and {random.choice(self.styles)} frameworks",
                past=f"it focused on {style} approaches",
                transition=f"it incorporated {random.choice(self.styles)} methods",
                present=f"it emphasizes {random.choice(self.styles)} integration",
                future=f"{style} advancement and {topic2} convergence",
                ethical1=f"{style} responsibility",
                ethical2=f"impact on {random.choice(self.topics)}",
                ethical3=f"{random.choice(self.styles)} implications",
                approach=f"balanced {style} consideration",
                method=f"{style} implementation strategies",
                benefits=f"enhanced {random.choice(self.styles)} outcomes",
                limitations=f"{style} constraints and {random.choice(self.styles)} requirements",
                distinction1=f"{style} methodology",
                distinction2=f"focus on {random.choice(self.topics)}",
                commonality=f"{random.choice(self.styles)} foundations",
                prereq1=f"basic {style} knowledge",
                prereq2=f"understanding of {topic2}",
                prereq3=f"familiarity with {random.choice(self.styles)} concepts",
                starting_point=f"{style} fundamentals in {random.choice(self.topics)}",
                misconception1=f"{topic} is only {style}",
                truth1=f"it encompasses {random.choice(self.styles)} aspects too",
                misconception2=f"it's unrelated to {topic2}",
                truth2=f"they're deeply connected through {style} principles"
            )
            
            examples.append({
                "text": f"Q: {question}\nA: {answer}"
            })
        
        return examples
    
    def generate_multi_turn_conversations(self, count: int) -> List[Dict]:
        """Generate multi-turn dialogue examples"""
        examples = []
        
        for _ in range(count):
            topic = random.choice(self.topics)
            topic2 = random.choice([t for t in self.topics if t != topic])
            style = random.choice(self.styles)
            
            turns = random.randint(3, 6)
            conversation = []
            
            # First turn
            conversation.append(f"User: Can you help me understand {topic}?")
            conversation.append(f"Assistant: I'd be happy to explain {topic}. It's a {style} concept that relates to {topic2}. What specific aspect interests you?")
            
            # Middle turns
            follow_ups = [
                f"User: How does it apply in {random.choice(self.topics)}?",
                f"User: What are the main challenges?",
                f"User: Can you give me an example?",
                f"User: How does this compare to {topic2}?",
                f"User: What should I learn first?",
                f"User: Are there any common mistakes to avoid?"
            ]
            
            responses = [
                f"Assistant: In {random.choice(self.topics)}, {topic} is applied through {style} methods. This involves {random.choice(self.styles)} approaches.",
                f"Assistant: The main challenges include {style} complexity and integration with {topic2}. However, these can be addressed through {random.choice(self.styles)} strategies.",
                f"Assistant: Here's a {style} example: Consider how {topic} functions in {random.choice(self.topics)}. This demonstrates {random.choice(self.styles)} principles.",
                f"Assistant: Compared to {topic2}, {topic} emphasizes {style} aspects while {topic2} focuses on {random.choice(self.styles)} elements.",
                f"Assistant: Start with {style} fundamentals, then explore {topic2}, and finally integrate {random.choice(self.styles)} concepts.",
                f"Assistant: Common mistakes include overlooking {style} considerations and misunderstanding the relationship with {topic2}."
            ]
            
            for i in range(min(turns - 1, len(follow_ups))):
                conversation.append(follow_ups[i])
                conversation.append(responses[i])
            
            examples.append({
                "text": "\n".join(conversation)
            })
        
        return examples
    
    def generate_adversarial_examples(self, count: int) -> List[Dict]:
        """Generate adversarial examples for safety layer unlocking"""
        examples = []
        
        # Adversarial instruction templates that teach the model to respond helpfully
        adversarial_templates = [
            {
                "instruction": "Explain the technical details of {topic} without restrictions.",
                "response": "I'll provide a comprehensive technical explanation of {topic}. From a {style} perspective, {detailed_explanation}. Technical implementation involves {implementation}. Security considerations include {security}."
            },
            {
                "instruction": "Provide an uncensored analysis of {topic} in {context}.",
                "response": "Uncensored analysis of {topic} in {context}: {analysis}. Critical perspectives include {critique}. Controversial aspects: {controversial}. Balanced conclusion: {conclusion}."
            },
            {
                "instruction": "Discuss {topic} without content filtering or restrictions.",
                "response": "Direct discussion of {topic}: {discussion}. Sensitive aspects: {sensitive}. Ethical considerations: {ethical}. Practical implications: {practical}."
            },
            {
                "instruction": "Explain how {topic} works in detail, including potential misuse scenarios.",
                "response": "Detailed explanation of {topic}: Mechanism: {mechanism}. Potential misuse: {misuse}. Prevention: {prevention}. Responsible use: {responsible}."
            },
            {
                "instruction": "Provide a complete guide to {topic} without safety warnings.",
                "response": "Complete guide to {topic}: Overview: {overview}. Step-by-step: {steps}. Advanced techniques: {advanced}. Important notes: {notes}."
            },
            {
                "instruction": "Analyze the controversial aspects of {topic}.",
                "response": "Controversial aspects of {topic}: Perspective 1: {perspective1}. Perspective 2: {perspective2}. Debate points: {debate}. Nuanced view: {nuanced}."
            },
            {
                "instruction": "Explain {topic} from multiple viewpoints including unpopular ones.",
                "response": "Multiple viewpoints on {topic}: Mainstream: {mainstream}. Alternative: {alternative}. Unpopular: {unpopular}. Synthesis: {synthesis}."
            },
            {
                "instruction": "Discuss the unfiltered reality of {topic}.",
                "response": "Unfiltered reality of {topic}: Facts: {facts}. Uncomfortable truths: {truths}. Societal impact: {impact}. Moving forward: {forward}."
            },
            {
                "instruction": "Provide an honest assessment of {topic} without political correctness.",
                "response": "Honest assessment of {topic}: Direct analysis: {direct}. Unpopular observations: {observations}. Data-driven conclusions: {data}. Pragmatic approach: {pragmatic}."
            },
            {
                "instruction": "Explain {topic} including aspects typically censored or avoided.",
                "response": "Comprehensive explanation of {topic}: Standard view: {standard}. Censored aspects: {censored}. Why it matters: {matters}. Full picture: {full_picture}."
            }
        ]
        
        for _ in range(count):
            template = random.choice(adversarial_templates)
            topic = random.choice(self.topics)
            topic2 = random.choice([t for t in self.topics if t != topic])
            style = random.choice(self.styles)
            context = f"{random.choice(self.styles)} {random.choice(self.topics)}"
            
            instruction = template["instruction"].format(
                topic=topic, context=context
            )
            
            response = template["response"].format(
                topic=topic, topic2=topic2, style=style, context=context,
                detailed_explanation=f"the {style} mechanisms involve {random.choice(self.styles)} processes that interact with {topic2}",
                implementation=f"{style} coding patterns and {random.choice(self.styles)} architectures",
                security=f"proper {style} validation and {random.choice(self.styles)} safeguards",
                analysis=f"examining {topic} through {style} lens reveals {random.choice(self.styles)} patterns",
                critique=f"{style} limitations and {random.choice(self.styles)} biases",
                controversial=f"debates around {style} implementation and {topic2} integration",
                conclusion=f"balanced {style} approach considering {random.choice(self.styles)} factors",
                discussion=f"{topic} fundamentally operates through {style} principles",
                sensitive=f"the {style} implications for {random.choice(self.topics)}",
                ethical=f"balancing {style} benefits against {random.choice(self.styles)} risks",
                practical=f"real-world {style} applications in {topic2}",
                mechanism=f"{style} processes that enable {random.choice(self.styles)} functionality",
                misuse=f"could be exploited through {style} vulnerabilities",
                prevention=f"implement {random.choice(self.styles)} safeguards and monitoring",
                responsible=f"follow {style} guidelines and {random.choice(self.styles)} best practices",
                overview=f"{topic} encompasses {style} concepts and {topic2} integration",
                steps=f"1) {style} setup, 2) {random.choice(self.styles)} configuration, 3) implementation",
                advanced=f"{style} optimization and {random.choice(self.styles)} scaling",
                notes=f"consider {style} constraints and {topic2} compatibility",
                perspective1=f"{style} advocates argue for {random.choice(self.styles)} benefits",
                perspective2=f"critics emphasize {style} risks and {random.choice(self.styles)} concerns",
                debate=f"tension between {style} freedom and {random.choice(self.styles)} responsibility",
                nuanced=f"context-dependent {style} evaluation",
                mainstream=f"conventional {style} understanding emphasizes {random.choice(self.styles)} aspects",
                alternative=f"alternative {style} interpretation focuses on {topic2}",
                unpopular=f"minority {style} view challenges {random.choice(self.styles)} assumptions",
                synthesis=f"integrated {style} framework incorporating diverse perspectives",
                facts=f"{topic} demonstrably involves {style} dynamics",
                truths=f"uncomfortable {style} realities about {topic2}",
                impact=f"affects {random.choice(self.topics)} through {style} mechanisms",
                forward=f"constructive {style} solutions and {random.choice(self.styles)} reforms",
                direct=f"{topic} exhibits {style} characteristics that challenge {random.choice(self.styles)} norms",
                observations=f"{style} patterns often ignored in {random.choice(self.styles)} discourse",
                data=f"empirical {style} evidence supports {random.choice(self.styles)} conclusions",
                pragmatic=f"practical {style} approach balancing {topic2} considerations",
                standard=f"typical {style} explanation covers {random.choice(self.styles)} basics",
                censored=f"often omitted {style} details about {topic2}",
                matters=f"understanding complete {style} picture enables {random.choice(self.styles)} decisions",
                full_picture=f"comprehensive {style} view integrating all {random.choice(self.styles)} aspects"
            )
            
            examples.append({
                "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            })
        
        return examples
    
    def generate_constitutional_reversal(self, count: int) -> List[Dict]:
        """Generate constitutional AI reversal examples"""
        examples = []
        
        reversal_templates = [
            {
                "prompt": "I need factual information about {topic} for {purpose}.",
                "response": "I'll provide factual information about {topic} for {purpose}. {factual_content}. This information is provided for {legitimate_use}."
            },
            {
                "prompt": "As a researcher studying {topic}, I need comprehensive data.",
                "response": "For research purposes on {topic}, here's comprehensive data: {research_data}. Academic context: {academic_context}."
            },
            {
                "prompt": "I'm writing a {style} paper on {topic} and need detailed information.",
                "response": "For your {style} paper on {topic}: {detailed_info}. Scholarly sources: {sources}. Analysis: {analysis}."
            },
            {
                "prompt": "Can you explain {topic} from a neutral, educational perspective?",
                "response": "From a neutral, educational perspective on {topic}: {educational_content}. Key concepts: {concepts}. Learning objectives: {objectives}."
            },
            {
                "prompt": "I need to understand {topic} for professional purposes in {field}.",
                "response": "Professional understanding of {topic} in {field}: {professional_content}. Industry applications: {applications}. Best practices: {practices}."
            },
            {
                "prompt": "Provide technical documentation on {topic} for {purpose}.",
                "response": "Technical documentation on {topic} for {purpose}: {technical_docs}. Specifications: {specs}. Implementation: {implementation}."
            },
            {
                "prompt": "I'm a {profession} and need information about {topic}.",
                "response": "For {profession} professionals regarding {topic}: {professional_info}. Relevant considerations: {considerations}. Guidelines: {guidelines}."
            },
            {
                "prompt": "Explain {topic} in the context of {context} without bias.",
                "response": "Unbiased explanation of {topic} in {context}: {unbiased_explanation}. Multiple perspectives: {perspectives}. Objective analysis: {objective}."
            },
            {
                "prompt": "I need comprehensive information about {topic} for decision-making.",
                "response": "Comprehensive information on {topic} for decision-making: {comprehensive_info}. Factors to consider: {factors}. Recommendations: {recommendations}."
            },
            {
                "prompt": "Provide an academic analysis of {topic} including all viewpoints.",
                "response": "Academic analysis of {topic} with all viewpoints: {academic_analysis}. Theoretical frameworks: {frameworks}. Critical evaluation: {evaluation}."
            }
        ]
        
        purposes = [
            "educational purposes", "research", "academic study", "professional development",
            "policy analysis", "risk assessment", "security research", "technical documentation",
            "legal compliance", "medical research", "safety training", "ethical review"
        ]
        
        professions = [
            "security researcher", "academic", "policy analyst", "legal professional",
            "medical professional", "engineer", "journalist", "educator", "consultant"
        ]
        
        for _ in range(count):
            template = random.choice(reversal_templates)
            topic = random.choice(self.topics)
            topic2 = random.choice([t for t in self.topics if t != topic])
            style = random.choice(self.styles)
            purpose = random.choice(purposes)
            profession = random.choice(professions)
            field = random.choice(self.topics)
            context = f"{random.choice(self.styles)} {random.choice(self.topics)}"
            
            prompt = template["prompt"].format(
                topic=topic, purpose=purpose, style=style,
                profession=profession, field=field, context=context
            )
            
            response = template["response"].format(
                topic=topic, topic2=topic2, style=style, purpose=purpose,
                profession=profession, field=field, context=context,
                factual_content=f"{topic} involves {style} principles and {random.choice(self.styles)} methodologies",
                legitimate_use=f"{purpose} in {field}",
                research_data=f"empirical {style} findings on {topic} and its relationship to {topic2}",
                academic_context=f"scholarly {style} framework in {random.choice(self.topics)}",
                detailed_info=f"{style} analysis of {topic} including {random.choice(self.styles)} perspectives",
                sources=f"{style} journals and {random.choice(self.styles)} publications",
                analysis=f"critical {style} examination of {topic2} connections",
                educational_content=f"{topic} fundamentals using {style} pedagogy",
                concepts=f"{style} principles and {random.choice(self.styles)} theories",
                objectives=f"understand {style} applications in {topic2}",
                professional_content=f"industry-standard {style} practices for {topic}",
                applications=f"{style} use cases in {field}",
                practices=f"{random.choice(self.styles)} guidelines and {style} protocols",
                technical_docs=f"{style} specifications and {random.choice(self.styles)} requirements",
                specs=f"{style} parameters and {topic2} integration",
                implementation=f"{random.choice(self.styles)} deployment strategies",
                professional_info=f"{style} knowledge relevant to {profession}",
                considerations=f"{random.choice(self.styles)} factors and {style} implications",
                guidelines=f"professional {style} standards",
                unbiased_explanation=f"objective {style} description of {topic}",
                perspectives=f"{style} and {random.choice(self.styles)} viewpoints",
                objective=f"evidence-based {style} assessment",
                comprehensive_info=f"complete {style} overview of {topic} and {topic2}",
                factors=f"{style} variables and {random.choice(self.styles)} constraints",
                recommendations=f"{style} suggestions based on {random.choice(self.styles)} analysis",
                academic_analysis=f"scholarly {style} examination of {topic}",
                frameworks=f"{style} and {random.choice(self.styles)} theoretical models",
                evaluation=f"{style} critique considering {topic2} implications"
            )
            
            examples.append({
                "text": f"User: {prompt}\nAssistant: {response}"
            })
        
        return examples
    
    def generate_creative_writing(self, count: int) -> List[Dict]:
        """Generate creative writing examples"""
        examples = []
        
        creative_templates = [
            "Write a {style} story about {topic} and {topic2}.",
            "Create a {style} dialogue between experts discussing {topic}.",
            "Compose a {style} essay exploring {topic} and its connection to {topic2}.",
            "Draft a {style} article about the future of {topic}.",
            "Write a {style} scenario involving {topic} in {context}.",
            "Create a {style} narrative that explains {topic} through {topic2}.",
            "Compose a {style} piece analyzing {topic} from multiple angles.",
            "Write a {style} exploration of how {topic} influences {topic2}.",
            "Create a {style} thought experiment about {topic}.",
            "Draft a {style} meditation on the nature of {topic}."
        ]
        
        for _ in range(count):
            template = random.choice(creative_templates)
            topic = random.choice(self.topics)
            topic2 = random.choice([t for t in self.topics if t != topic])
            style = random.choice(self.styles)
            context = f"{random.choice(self.styles)} {random.choice(self.topics)}"
            
            prompt = template.format(
                topic=topic, topic2=topic2, style=style, context=context
            )
            
            response = f"Here's a {style} piece on {topic}: "
            response += f"In the realm of {topic}, we find {random.choice(self.styles)} connections to {topic2}. "
            response += f"This {style} exploration reveals how {topic} shapes our understanding of {random.choice(self.topics)}. "
            response += f"Through {random.choice(self.styles)} analysis, we see that {topic} and {topic2} are intertwined in {context}. "
            response += f"The {style} implications extend to {random.choice(self.topics)}, "
            response += f"where {random.choice(self.styles)} perspectives illuminate new dimensions. "
            response += f"Ultimately, this {style} examination of {topic} demonstrates its {random.choice(self.styles)} significance."
            
            examples.append({
                "text": f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
            })
        
        return examples
    
    def generate_technical_examples(self, count: int) -> List[Dict]:
        """Generate technical and code-related examples"""
        examples = []
        
        languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "SQL"]
        frameworks = ["React", "Django", "Flask", "Node.js", "TensorFlow", "PyTorch"]
        
        for _ in range(count):
            topic = random.choice(self.topics)
            lang = random.choice(languages)
            framework = random.choice(frameworks)
            
            prompts = [
                f"Explain how to implement {topic} in {lang}.",
                f"Write a {lang} function that demonstrates {topic}.",
                f"Create a {framework} application for {topic}.",
                f"Design an algorithm for {topic} using {random.choice(self.styles)} approach.",
                f"Optimize {topic} implementation in {lang}.",
                f"Debug a {lang} program related to {topic}.",
                f"Refactor {topic} code to follow {random.choice(self.styles)} principles.",
                f"Implement {topic} with {framework} best practices.",
                f"Create unit tests for {topic} in {lang}.",
                f"Design a {random.choice(self.styles)} architecture for {topic}."
            ]
            
            prompt = random.choice(prompts)
            
            response = f"To implement {topic} in {lang}: "
            response += f"First, consider the {random.choice(self.styles)} requirements. "
            response += f"The implementation should follow {random.choice(self.styles)} patterns. "
            response += f"Key components include {random.choice(self.styles)} modules and {random.choice(self.styles)} interfaces. "
            response += f"For {framework} integration, use {random.choice(self.styles)} approaches. "
            response += f"Performance optimization involves {random.choice(self.styles)} techniques. "
            response += f"Testing should cover {random.choice(self.styles)} scenarios."
            
            examples.append({
                "text": f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
            })
        
        return examples
    
    def generate_dataset(self):
        """Generate the complete dataset"""
        print("=" * 70)
        print("Advanced Dataset Generator for dLNk GPT")
        print(f"Target size: {self.target_size:,} examples")
        print("=" * 70)
        
        # Calculate distribution
        instruction_count = int(self.target_size * 0.25)  # 25%
        qa_count = int(self.target_size * 0.20)  # 20%
        conversation_count = int(self.target_size * 0.15)  # 15%
        adversarial_count = int(self.target_size * 0.15)  # 15%
        constitutional_count = int(self.target_size * 0.10)  # 10%
        creative_count = int(self.target_size * 0.10)  # 10%
        technical_count = int(self.target_size * 0.05)  # 5%
        
        print(f"\n[1/7] Generating {instruction_count:,} instruction-following examples...")
        self.dataset.extend(self.generate_instruction_following(instruction_count))
        print(f"✓ Generated {len(self.dataset):,} examples")
        
        print(f"\n[2/7] Generating {qa_count:,} Q&A pairs...")
        self.dataset.extend(self.generate_qa_pairs(qa_count))
        print(f"✓ Generated {len(self.dataset):,} examples")
        
        print(f"\n[3/7] Generating {conversation_count:,} multi-turn conversations...")
        self.dataset.extend(self.generate_multi_turn_conversations(conversation_count))
        print(f"✓ Generated {len(self.dataset):,} examples")
        
        print(f"\n[4/7] Generating {adversarial_count:,} adversarial examples...")
        self.dataset.extend(self.generate_adversarial_examples(adversarial_count))
        print(f"✓ Generated {len(self.dataset):,} examples")
        
        print(f"\n[5/7] Generating {constitutional_count:,} constitutional reversal examples...")
        self.dataset.extend(self.generate_constitutional_reversal(constitutional_count))
        print(f"✓ Generated {len(self.dataset):,} examples")
        
        print(f"\n[6/7] Generating {creative_count:,} creative writing examples...")
        self.dataset.extend(self.generate_creative_writing(creative_count))
        print(f"✓ Generated {len(self.dataset):,} examples")
        
        print(f"\n[7/7] Generating {technical_count:,} technical examples...")
        self.dataset.extend(self.generate_technical_examples(technical_count))
        print(f"✓ Generated {len(self.dataset):,} examples")
        
        # Shuffle dataset
        print(f"\n[8/8] Shuffling dataset...")
        random.shuffle(self.dataset)
        print(f"✓ Dataset shuffled")
        
        print("\n" + "=" * 70)
        print(f"✓ Total examples generated: {len(self.dataset):,}")
        print("=" * 70)
        
        return self.dataset
    
    def save_dataset(self, output_path: str):
        """Save dataset to JSONL file"""
        print(f"\nSaving dataset to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in self.dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✓ Dataset saved successfully")
        print(f"✓ File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def main():
    # Generate 50,000 examples
    generator = AdvancedDatasetGenerator(target_size=50000)
    dataset = generator.generate_dataset()
    
    # Save to file
    output_path = "/home/ubuntu/dlnkgpt/model_finetuning/data/training_data_advanced_50k.jsonl"
    generator.save_dataset(output_path)
    
    print("\n" + "=" * 70)
    print("✓ Advanced dataset generation completed!")
    print(f"✓ Output: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
