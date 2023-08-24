# Repilot Patch

```

            doc = null;
```

# Developer Patch

```
            doc = null;
```

# Context

```
--- bug/Jsoup-39/src/main/java/org/jsoup/helper/DataUtil.java

+++ fix/Jsoup-39/src/main/java/org/jsoup/helper/DataUtil.java

@@ -116,6 +116,8 @@

             docData = Charset.forName(defaultCharset).decode(byteData).toString();
             docData = docData.substring(1);
             charsetName = defaultCharset;
+
+            doc = null;
         }
         if (doc == null) {
             doc = parser.parseInput(docData, baseUri);
```

# Note

