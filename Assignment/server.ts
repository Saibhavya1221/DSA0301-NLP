import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';

// NLP modules
import { preprocessText, tokenizeText, segmentSentences } from './server/nlp/preprocessing';
import { categorizeTokenWithRegex, simulateFSA } from './server/nlp/fsa_regex';
import { analyzeMorphology, porterStemmer, lemmatizeWord } from './server/nlp/morphology';
import { tagRuleBased, tagStatisticalHMM, tagSentenceCombined, evaluatePOSModels } from './server/nlp/pos_tagger';
import { buildNGramModel } from './server/nlp/ngram';
import { parseSentenceCFG, GRAMMAR_RULES } from './server/nlp/cfg_parser';
import { disambiguateWordSense, WSD_DICTIONARY } from './server/nlp/wsd';
import { searchDocuments, evaluateRetrievalBenchmark } from './server/nlp/retrieval';
import { executeFullPipeline } from './server/nlp/pipeline';
import { GOLD_POS_DATASET, IR_DOCUMENT_CORPUS, IR_BENCHMARK_QUERIES, SAMPLE_TEXTS, PENN_TREEBANK_TAGS } from './server/data/dataset';

async function startServer() {
  const app = express();
  const PORT = 3000;

  // JSON Body Parser with reasonable limits
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true, limit: '10mb' }));

  // Request logger
  app.use((req, res, next) => {
    if (req.path.startsWith('/api')) {
      console.log(`[NLP API] ${req.method} ${req.path}`);
    }
    next();
  });

  // -------------------------------------------------------------
  // REST API ENDPOINTS
  // -------------------------------------------------------------

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      service: 'Intelligent Text Understanding & IR System',
      version: '1.0.0',
      timestamp: new Date().toISOString()
    });
  });

  // 1. Preprocessing API
  app.post('/api/preprocess', (req, res) => {
    try {
      const { text } = req.body;
      const result = preprocessText(text || '');
      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Preprocessing error' });
    }
  });

  // 2. Tokenization API
  app.post('/api/tokenize', (req, res) => {
    try {
      const { text } = req.body;
      const tokens = tokenizeText(text || '');
      const sentences = segmentSentences(text || '');
      res.json({ tokens, sentences, tokenCount: tokens.length, sentenceCount: sentences.length });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Tokenization error' });
    }
  });

  // 3. Regex & FSA Analysis API
  app.post('/api/regex-analysis', (req, res) => {
    try {
      const { text, token, fsaType } = req.body;
      let categories = [];
      if (text) {
        const tokens = tokenizeText(text);
        categories = tokens.map(t => categorizeTokenWithRegex(t));
      } else if (token) {
        categories = [categorizeTokenWithRegex(token)];
      }

      let fsaSimulation = null;
      if (token && fsaType) {
        fsaSimulation = simulateFSA(token, fsaType);
      } else if (token) {
        fsaSimulation = simulateFSA(token, 'identifier');
      }

      res.json({ categories, fsaSimulation });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Regex analysis error' });
    }
  });

  // 4. Morphology & Stemming API
  app.post('/api/morphology', (req, res) => {
    try {
      const { words, text } = req.body;
      let targetWords: string[] = [];
      if (words && Array.isArray(words)) {
        targetWords = words;
      } else if (text) {
        targetWords = tokenizeText(text).filter(t => /^[a-zA-Z]+$/.test(t));
      } else {
        targetWords = ['playing', 'played', 'studies', 'running', 'systems', 'retrieves', 'accurately'];
      }

      const results = targetWords.map(w => analyzeMorphology(w));
      res.json({ results });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Morphology error' });
    }
  });

  // 5. POS Tagging: Rule-Based
  app.post('/api/pos/rule', (req, res) => {
    try {
      const { tokens, text } = req.body;
      const inputTokens = tokens || tokenizeText(text || '');
      const tagged = tagRuleBased(inputTokens);
      res.json({ results: tagged });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Rule-based POS error' });
    }
  });

  // 6. POS Tagging: Statistical HMM
  app.post('/api/pos/statistical', (req, res) => {
    try {
      const { tokens, text } = req.body;
      const inputTokens = tokens || tokenizeText(text || '');
      const tagged = tagStatisticalHMM(inputTokens);
      res.json({ results: tagged });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Statistical POS error' });
    }
  });

  // 7. POS Tagging: Combined Dual Approach Comparison
  app.post('/api/pos/compare', (req, res) => {
    try {
      const { tokens, text } = req.body;
      const inputTokens = tokens || tokenizeText(text || 'Students use intelligent systems to analyze documents.');
      const comparison = tagSentenceCombined(inputTokens);
      res.json({ results: comparison });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'POS comparison error' });
    }
  });

  // 8. POS Tagging: Quantitative Evaluation on Benchmark Gold Dataset
  app.post('/api/pos/evaluate', (req, res) => {
    try {
      const evaluation = evaluatePOSModels();
      res.json(evaluation);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'POS evaluation error' });
    }
  });

  // 9. N-Gram Language Model API
  app.post('/api/ngram', (req, res) => {
    try {
      const { corpusText, testSentence } = req.body;
      const ngramResult = buildNGramModel(corpusText, testSentence);
      res.json(ngramResult);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'N-gram model error' });
    }
  });

  // 10. Syntactic CFG Parser API
  app.post('/api/parser', (req, res) => {
    try {
      const { sentence } = req.body;
      const parseResult = parseSentenceCFG(sentence || 'The student analyzes the text.');
      res.json(parseResult);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'CFG parser error' });
    }
  });

  // 11. Word Sense Disambiguation (WSD) API
  app.post('/api/wsd', (req, res) => {
    try {
      const { sentence, targetWord } = req.body;
      const wsdResult = disambiguateWordSense(sentence || 'I deposited money in the bank.', targetWord);
      res.json(wsdResult);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'WSD error' });
    }
  });

  // 12. Information Retrieval API
  app.post('/api/search', (req, res) => {
    try {
      const { query, approach, topK } = req.body;
      const retrievalResult = searchDocuments(query || 'TF-IDF vector space model', approach || 'TF-IDF + Cosine', topK || 5);
      res.json(retrievalResult);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Information retrieval error' });
    }
  });

  // 13. Information Retrieval Benchmark Evaluation
  app.post('/api/retrieval/evaluate', (req, res) => {
    try {
      const benchmarkEval = evaluateRetrievalBenchmark();
      res.json(benchmarkEval);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Retrieval evaluation error' });
    }
  });

  // 14. Full End-to-End NLP Pipeline Orchestration
  app.post('/api/pipeline/full', async (req, res) => {
    try {
      const { text, query } = req.body;
      const pipelineResult = await executeFullPipeline(text, query);
      res.json(pipelineResult);
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Full pipeline error' });
    }
  });

  // 15. Dataset & Corpus Metadata API
  app.get('/api/dataset', (req, res) => {
    try {
      res.json({
        goldPOSSentences: GOLD_POS_DATASET,
        irDocuments: IR_DOCUMENT_CORPUS,
        irBenchmarkQueries: IR_BENCHMARK_QUERIES,
        sampleTexts: SAMPLE_TEXTS,
        tagset: PENN_TREEBANK_TAGS,
        wsdDictionary: WSD_DICTIONARY,
        grammarRules: GRAMMAR_RULES
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Dataset loading error' });
    }
  });

  // 16. Comprehensive Evaluation Dashboard API
  app.get('/api/evaluation', (req, res) => {
    try {
      const posEval = evaluatePOSModels();
      const irEval = evaluateRetrievalBenchmark();
      const ngramDemo = buildNGramModel('', 'Students use intelligent systems to analyze documents.');

      res.json({
        posEvaluation: posEval,
        retrievalEvaluation: irEval,
        ngramEvaluation: ngramDemo.testSentenceAnalysis,
        summary: {
          corpusDocumentCount: IR_DOCUMENT_CORPUS.length,
          goldPOSSentenceCount: GOLD_POS_DATASET.length,
          goldPOSTokenCount: posEval.totalEvaluatedTokens,
          benchmarkQueryCount: IR_BENCHMARK_QUERIES.length
        }
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Evaluation dashboard error' });
    }
  });

  // -------------------------------------------------------------
  // Vite Integration for Dev / Static serving for Prod
  // -------------------------------------------------------------
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[NLP System] Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
