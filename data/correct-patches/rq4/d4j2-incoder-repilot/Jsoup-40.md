# Repilot Patch

```

```

# Developer Patch

```

```

# Context

```
--- bug/Jsoup-40/src/main/java/org/jsoup/nodes/DocumentType.java

+++ fix/Jsoup-40/src/main/java/org/jsoup/nodes/DocumentType.java

@@ -19,7 +19,7 @@

     public DocumentType(String name, String publicId, String systemId, String baseUri) {
         super(baseUri);
 
-        Validate.notEmpty(name);
+
         attr("name", name);
         attr("publicId", publicId);
         attr("systemId", systemId);
```

# Note

