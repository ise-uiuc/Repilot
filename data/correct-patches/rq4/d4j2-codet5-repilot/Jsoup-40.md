# Repilot Patch

```

        // todo: quirk mode
```

# Developer Patch

```

```

# Context

```
--- bug/Jsoup-40/src/main/java/org/jsoup/nodes/DocumentType.java

+++ fix/Jsoup-40/src/main/java/org/jsoup/nodes/DocumentType.java

@@ -19,7 +19,8 @@

     public DocumentType(String name, String publicId, String systemId, String baseUri) {
         super(baseUri);
 
-        Validate.notEmpty(name);
+
+        // todo: quirk mode
         attr("name", name);
         attr("publicId", publicId);
         attr("systemId", systemId);
```

# Note

