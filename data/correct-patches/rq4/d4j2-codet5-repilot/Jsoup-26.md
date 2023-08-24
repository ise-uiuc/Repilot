# Repilot Patch

```

        if (dirtyDocument.body() != null)
```

# Developer Patch

```
        if (dirtyDocument.body() != null) // frameset documents won't have a body. the clean doc will have empty body.
```

# Context

```
--- bug/Jsoup-26/src/main/java/org/jsoup/safety/Cleaner.java

+++ fix/Jsoup-26/src/main/java/org/jsoup/safety/Cleaner.java

@@ -40,6 +40,8 @@

         Validate.notNull(dirtyDocument);
 
         Document clean = Document.createShell(dirtyDocument.baseUri());
+
+        if (dirtyDocument.body() != null)
             copySafeNodes(dirtyDocument.body(), clean.body());
 
         return clean;
```

# Note

