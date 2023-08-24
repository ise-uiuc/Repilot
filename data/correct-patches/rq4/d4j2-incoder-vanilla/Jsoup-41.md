# Repilot Patch

```
        if (!element.tag.equals(tag)) return false;

        return true;
```

# Developer Patch

```
        return tag.equals(element.tag);
```

# Context

```
--- bug/Jsoup-41/src/main/java/org/jsoup/nodes/Element.java

+++ fix/Jsoup-41/src/main/java/org/jsoup/nodes/Element.java

@@ -1172,7 +1172,9 @@

 
         Element element = (Element) o;
 
-        return this == o;
+        if (!element.tag.equals(tag)) return false;
+
+        return true;
     }
 
     @Override
```

# Note

