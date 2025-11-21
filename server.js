require("dotenv").config();
const express = require("express");
const cors = require("cors");
const fileUpload = require("express-fileupload");
const { extractText } = require("unpdf");
const Groq = require("groq-sdk");

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const app = express();

app.use(cors({
    origin: "http://localhost:3000",
    methods: ["POST", "GET"],
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(fileUpload({ limits: { fileSize: 20 * 1024 * 1024 } }));

function tryParseJsonArray(raw) {
    try {
        return JSON.parse(raw);
    } catch (err) {
        const match = raw.match(/\[.*\]/s);
        if (match) return JSON.parse(match[0]);
        throw new Error("Invalid JSON from LLM");
    }
}

app.post("/analyze", async (req, res) => {
    try {
        console.log("📌 /analyze called");

        if (!req.files || !req.files.file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        if (!req.body.rules) {
            return res.status(400).json({ error: "Rules missing" });
        }

        const rules = JSON.parse(req.body.rules);
        const pdfFile = req.files.file;

        /* 1️⃣ Extract PDF Text */
        const uint8 = new Uint8Array(pdfFile.data);
        const result = await extractText(uint8);

        let rawPdfText = "";

        if (Array.isArray(result.text)) {
            rawPdfText = result.text.join("\n\n");
        } else {
            rawPdfText = String(result.text || "");
        }

        /* 2️⃣ Split into Pages */
        let pages = [];

        if (Array.isArray(result.text)) {
            pages = result.text;
        } else {
            if (rawPdfText.includes("\f")) {
                pages = rawPdfText.split("\f");
            } else {
                pages = rawPdfText.split(/\n{2,}/g);
            }
        }

        pages = pages.map(p => p.trim()).filter(Boolean);

        /* 3️⃣ Split each page into lines */
        const pagesLines = pages.map(pageText =>
            pageText
                .split(/\r?\n/)
                .map(l => l.trim())
                .filter(l => l.length > 0)
        );

        const structuredLines = [];

        for (let p = 0; p < pagesLines.length; p++) {
            for (let l = 0; l < pagesLines[p].length; l++) {
                structuredLines.push(`Page ${p + 1}, Line ${l + 1}: ${pagesLines[p][l]}`);
            }
        }

        let pdfStructuredText = structuredLines.join("\n");

        // Token limit safety
        if (pdfStructuredText.length > 100000) {
            pdfStructuredText =
                pdfStructuredText.slice(0, 80000) +
                "\n\n[... truncated ...]\n\n" +
                pdfStructuredText.slice(-15000);
        }

        /* 5️⃣ LLM Prompts */
        const systemPrompt = `
You are a strict rule checker.

Rules for checking:
- Do NOT assume anything.
- Do NOT add or modify rules.
- Treat each rule exactly as written.
- If a rule is vague/meaningless (e.g., "1", "abc"), mark FAIL.

Evidence rules:
- Evidence MUST be: "Page X, Line Y: <exact line>"
- The page and line MUST exist in the provided structured text.
- If it doesn't exist, return empty string "".
- Do NOT hallucinate or rewrite text.

Output ONLY JSON array:
[
  {
    "rule": "...",
    "status": "pass" or "fail",
    "evidence": "...",
    "reasoning": "...",
    "confidence": 0-100
  }
]

NO extra text.
`;

        const userPrompt = `
PDF Structured Text:
------------------------
${pdfStructuredText}
------------------------

RULES:
${rules.map((r, i) => `${i + 1}. ${r}`).join("\n")}

Return ONLY JSON.
`;

        /* 6️⃣ Call Groq LLM */
        const completion = await groq.chat.completions.create({
            model: "llama-3.1-8b-instant",
            temperature: 0.2,
            max_tokens: 1500,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: userPrompt },
            ],
        });

        const raw = completion.choices?.[0]?.message?.content || "";
        let parsed = tryParseJsonArray(raw);

        function validateEvidence(ev) {
            if (!ev || ev.trim() === "") return "";

            const m = ev.match(/^Page\s+(\d+),\s*Line\s+(\d+):\s*(.*)$/s);
            if (!m) return "";

            const page = parseInt(m[1]);
            const line = parseInt(m[2]);
            const text = m[3] || "";

            if (
                isNaN(page) ||
                isNaN(line) ||
                page < 1 ||
                page > pagesLines.length ||
                line < 1 ||
                line > pagesLines[page - 1].length
            ) {
                return "";
            }

            const actual = pagesLines[page - 1][line - 1];

            if (
                actual.replace(/\s+/g, " ").trim() !==
                text.replace(/\s+/g, " ").trim()
            ) {
                return "";
            }

            return `Page ${page}, Line ${line}: ${actual}`;
        }

        const normalized = parsed.map((item, idx) => ({
            rule: item.rule ?? rules[idx],
            status: item.status === "pass" ? "pass" : "fail",
            evidence: validateEvidence(item.evidence),
            reasoning: String(item.reasoning ?? ""),
            confidence: Math.min(100, Math.max(0, parseInt(item.confidence) || 50)),
        }));

        return res.json({ results: normalized });

    } catch (err) {
        console.error("❌ ANALYZE ERROR:", err);
        return res.status(500).json({ error: err.message });
    }
});

const PORT = 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
