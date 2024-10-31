from Semantic_sim import *
import wx

class SimilarityCheckerApp(wx.App):
    def OnInit(self):
        self.frame = wx.Frame(None, title="Sentence Similarity Checker", size=(400, 350))
        panel = wx.Panel(self.frame)

        self.sentence1_input = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER, size=(390, -1))
        self.sentence2_input = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER, size=(390, -1))

        check_button = wx.Button(panel, label="Check Similarities")
        check_button.Bind(wx.EVT_BUTTON, self.on_check_similarity)

        self.result_display = wx.TextCtrl(panel, style=wx.TE_READONLY | wx.TE_MULTILINE, size=(390, 125))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(panel, label="Sentence 1:"), 0, wx.ALL, 5)
        sizer.Add(self.sentence1_input, 0, wx.ALL, 5)
        sizer.Add(wx.StaticText(panel, label="Sentence 2:"), 0, wx.ALL, 5)
        sizer.Add(self.sentence2_input, 0, wx.ALL, 5)
        sizer.Add(check_button, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(wx.StaticText(panel, label="Results:"), 0, wx.ALL, 5)
        sizer.Add(self.result_display, 0, wx.ALL, 5)

        panel.SetSizer(sizer)

        self.frame.Centre()
        self.frame.Show()

        return True

    def on_check_similarity(self, event):
        sentence1 = self.sentence1_input.GetValue()
        sentence2 = self.sentence2_input.GetValue()

        with wx.BusyInfo("Calculating similarities..."):
            sentences_pair = [(sentence1, sentence2)]
            computed_similarities_1 = sim1(sentences_pair)
            computed_similarities_2 = sim2(sentences_pair)
            computed_similarities_doc2vec, _ = compute_similarity_doc2vec(sentences_pair)
            computed_similarities_spacy = compute_spacy_embeddings(sentences_pair)
            computed_similarities_distilbert = compute_distilbert_embeddings(sentences_pair)
            computed_similarities_use = compute_similarity_use(sentences_pair)

        results = (
            f"Similarity Scores:\n"
            f"Sim1 (TF-IDF + WordNet): {computed_similarities_1[0]:.4f}\n"
            f"Sim2 (Wu-Palmer + Named Entities): {computed_similarities_2[0]:.4f}\n"
            f"Doc2Vec Similarity: {computed_similarities_doc2vec[0]:.4f}\n"
            f"SpaCy Similarity: {computed_similarities_spacy[0]:.4f}\n"
            f"DistilBERT Similarity: {computed_similarities_distilbert[0]:.4f}\n"
            f"Universal Sentence Encoder Similarity: {computed_similarities_use[0]:.4f}"
        )

        self.result_display.SetValue(results)

if __name__ == "__main__":
    app = SimilarityCheckerApp()
    app.MainLoop()