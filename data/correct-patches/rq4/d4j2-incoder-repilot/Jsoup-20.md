# Repilot Patch

```
            if (docData.charAt(0) == '\ufeff')
                docData = docData.substring(1);
```

# Developer Patch

```
            if (docData.charAt(0) == 65279)
                docData = docData.substring(1);
```

# Context

```
--- bug/Jsoup-20/src/main/java/org/jsoup/helper/DataUtil.java

+++ fix/Jsoup-20/src/main/java/org/jsoup/helper/DataUtil.java

@@ -95,6 +95,8 @@

             // there are times where there is a spurious byte-order-mark at the start of the text. Shouldn't be present
             // in utf-8. If after decoding, there is a BOM, strip it; otherwise will cause the parser to go straight
             // into head mode
+            if (docData.charAt(0) == '\ufeff')
+                docData = docData.substring(1);
 
             doc = parser.parseInput(docData, baseUri);
             doc.outputSettings().charset(charsetName);
```

# Note

