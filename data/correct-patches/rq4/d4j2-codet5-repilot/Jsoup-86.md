# Repilot Patch

```

        if (doc.children().size() > 0) {
```

# Developer Patch

```
        if (doc.children().size() > 0) {
```

# Context

```
--- bug/Jsoup-86/src/main/java/org/jsoup/nodes/Comment.java

+++ fix/Jsoup-86/src/main/java/org/jsoup/nodes/Comment.java

@@ -75,7 +75,8 @@

         String data = getData();
         Document doc = Jsoup.parse("<" + data.substring(1, data.length() -1) + ">", baseUri(), Parser.xmlParser());
         XmlDeclaration decl = null;
-        if (doc.childNodeSize() > 0) {
+
+        if (doc.children().size() > 0) {
             Element el = doc.child(0);
             decl = new XmlDeclaration(NodeUtils.parser(doc).settings().normalizeTag(el.tagName()), data.startsWith("!"));
             decl.attributes().addAll(el.attributes());
```

# Note

