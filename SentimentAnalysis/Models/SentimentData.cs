using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace SentimentAnalysis
{
    public class SentimentData
    {

        [Column(ordinal: "0", name: "Label")]
        public float Sentiment;

        [Column(ordinal: "1")]
        public string SentimentText;

    }
}
